"""
CoachCheck — AI Coaching Hallucination Detector

Validates that CoachMe+'s coaching feedback is actually grounded in
what's visible in the video. Cross-checks AI coaching claims with
independent Pegasus verification passes.

Uses: Twelve Labs Pegasus 1.2 (via client.analyze())
"""

import re
import time


def _analyze_with_retry(client, prompt, video_id, max_retries=5):
    """Call client.analyze() with automatic retry on rate-limit (429)."""
    for attempt in range(max_retries):
        try:
            return client.analyze(prompt=prompt, video_id=video_id)
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "too_many_requests" in err_str or "rate" in err_str:
                # Parse retry-after if available, else exponential backoff
                wait = min(10 * (2 ** attempt), 60)
                import re as _re
                match = _re.search(r"retry-after.*?(\d+)", str(e))
                if match:
                    wait = int(match.group(1)) + 1
                print(f"  Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after {max_retries} retries due to rate limiting")


def parse_claims(raw_text):
    """Parse Pegasus claim-extraction output into structured list.

    Expected format from Pegasus:
        CLAIM 1: <claim text> | TIMESTAMP: <time> | ABOUT: <body part>
        CLAIM 2: ...

    Returns list of dicts with keys: text, timestamp, about.
    """
    claims = []
    pattern = re.compile(
        r"CLAIM\s*\d+\s*:\s*(.+?)\s*\|\s*TIMESTAMP\s*:\s*(.+?)\s*\|\s*ABOUT\s*:\s*(.+)",
        re.IGNORECASE,
    )
    for line in raw_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if match:
            claims.append({
                "text": match.group(1).strip(),
                "timestamp": match.group(2).strip(),
                "about": match.group(3).strip(),
            })
    # Fallback: if structured parsing found nothing, try a looser split
    if not claims:
        claims = _fallback_parse_claims(raw_text)
    return claims


def _fallback_parse_claims(raw_text):
    """Looser parser when Pegasus doesn't follow the exact format."""
    claims = []
    # Try to find numbered lines with any structure
    for line in raw_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Match lines starting with a number or bullet
        if re.match(r"^(\d+[\.\):]|[-*])\s+", line):
            text = re.sub(r"^(\d+[\.\):]|[-*])\s+", "", line).strip()
            # Try to extract a timestamp like "0:04", "0:04-0:06", "4s", etc.
            ts_match = re.search(
                r"(\d+:\d{2}(?:\s*[-–]\s*\d+:\d{2})?|\d+\.?\d*\s*s(?:ec)?)",
                text,
            )
            timestamp = ts_match.group(1) if ts_match else "unknown"
            claims.append({
                "text": text,
                "timestamp": timestamp,
                "about": "technique",
            })
    return claims


def parse_verification(raw_text):
    """Parse Pegasus verification response.

    Expected format:
        ACCURATE: yes/no
        WHAT I SEE: <observation>
        CONFIDENCE: high/medium/low
    """
    result = {
        "accurate": False,
        "what_i_see": raw_text.strip(),
        "confidence": "low",
    }

    accurate_match = re.search(
        r"ACCURATE\s*:\s*(yes|no|true|false)", raw_text, re.IGNORECASE
    )
    if accurate_match:
        val = accurate_match.group(1).lower()
        result["accurate"] = val in ("yes", "true")

    see_match = re.search(
        r"WHAT I SEE\s*:\s*(.+?)(?:\n|CONFIDENCE|$)",
        raw_text,
        re.IGNORECASE | re.DOTALL,
    )
    if see_match:
        result["what_i_see"] = see_match.group(1).strip()

    conf_match = re.search(
        r"CONFIDENCE\s*:\s*(high|medium|low)", raw_text, re.IGNORECASE
    )
    if conf_match:
        result["confidence"] = conf_match.group(1).lower()

    return result


def validate_coaching(athlete_sample, client):
    """Validate coaching feedback claims against the actual video.

    Args:
        athlete_sample: FiftyOne sample with 'coaching_feedback' and
            'tl_video_id' fields already populated by CoachMe Core.
        client: Initialized TwelveLabs client.

    Returns:
        dict with keys:
            claims        – list of per-claim validation dicts
            grounding_score – percentage of claims verified as grounded
            hallucinations_flagged – count of ungrounded claims
    """
    coaching_text = athlete_sample["coaching_feedback"]
    video_id = athlete_sample["tl_video_id"]

    if not coaching_text or not video_id:
        return {
            "claims": [],
            "grounding_score": 0.0,
            "hallucinations_flagged": 0,
        }

    # ── Step 1: Extract specific factual claims from coaching feedback ──
    extraction_prompt = (
        "From the coaching feedback below, extract each specific factual "
        "claim about the athlete's technique. For each claim, provide:\n"
        "- The exact claim made\n"
        "- The timestamp referenced (if any, otherwise write 'general')\n"
        "- What body part or movement it is about\n\n"
        f"Coaching feedback:\n{coaching_text}\n\n"
        "Format EACH claim on its own line exactly as:\n"
        "CLAIM 1: <claim text> | TIMESTAMP: <time> | ABOUT: <body part or movement>\n"
        "CLAIM 2: <claim text> | TIMESTAMP: <time> | ABOUT: <body part or movement>\n"
        "... and so on."
    )

    claims_response = _analyze_with_retry(client, extraction_prompt, video_id)
    claims = parse_claims(claims_response.data)

    if not claims:
        return {
            "claims": [],
            "grounding_score": 100.0,
            "hallucinations_flagged": 0,
        }

    # ── Step 2: Independently verify each claim via Pegasus ──
    validated_claims = []
    for claim in claims:
        verification_prompt = (
            f"Look at this video at approximately {claim['timestamp']}.\n\n"
            f"Describe exactly what you see regarding {claim['about']}. "
            "Be very specific about position, angle, and movement.\n\n"
            "Then answer: Does the following statement accurately describe "
            "what is happening?\n\n"
            f'Statement: "{claim["text"]}"\n\n'
            "Answer with EXACTLY this format:\n"
            "ACCURATE: yes or no\n"
            "WHAT I SEE: <your independent observation>\n"
            "CONFIDENCE: high, medium, or low"
        )

        verification_response = _analyze_with_retry(client, verification_prompt, video_id)

        parsed = parse_verification(verification_response.data)
        validated_claims.append({
            "original_claim": claim["text"],
            "timestamp": claim["timestamp"],
            "about": claim["about"],
            "verification_response": parsed["what_i_see"],
            "is_grounded": parsed["accurate"],
            "confidence": parsed["confidence"],
        })

    # ── Step 3: Compute aggregate scores ──
    grounded_count = sum(1 for c in validated_claims if c["is_grounded"])
    total = len(validated_claims)

    return {
        "claims": validated_claims,
        "grounding_score": round(grounded_count / total * 100, 1) if total else 0.0,
        "hallucinations_flagged": total - grounded_count,
    }
