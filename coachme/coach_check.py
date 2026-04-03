"""
CoachCheck — AI Coaching Hallucination Detector

Validates that coaching feedback is actually grounded in what's visible
in the video. Cross-checks each claim with an independent Pegasus pass.

Powered by: Pegasus (claim extraction + independent verification)
"""


def parse_claims(raw_text):
    """Parse extracted claims from Pegasus output."""
    claims = []
    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if not line or not line.upper().startswith("CLAIM"):
            continue

        # Format: CLAIM N: [text] | TIMESTAMP: [time] | ABOUT: [body part]
        parts = line.split("|")
        claim_text = parts[0].split(":", 1)[1].strip() if ":" in parts[0] else parts[0]
        timestamp = ""
        about = ""

        for part in parts[1:]:
            part = part.strip()
            if part.upper().startswith("TIMESTAMP:"):
                timestamp = part.split(":", 1)[1].strip()
            elif part.upper().startswith("ABOUT:"):
                about = part.split(":", 1)[1].strip()

        if claim_text:
            claims.append({
                "text": claim_text,
                "timestamp": timestamp or "0:00",
                "about": about or "technique",
            })

    return claims


def parse_verification(raw_text):
    """Parse verification response from Pegasus."""
    result = {
        "accurate": False,
        "what_i_see": raw_text.strip(),
        "confidence": "medium",
    }

    for line in raw_text.strip().split("\n"):
        line = line.strip().upper()
        if line.startswith("ACCURATE:"):
            val = line.split(":", 1)[1].strip().lower()
            result["accurate"] = val in ("yes", "true", "correct", "confirmed")
        elif line.startswith("WHAT I SEE:"):
            result["what_i_see"] = line.split(":", 1)[1].strip()
        elif line.startswith("CONFIDENCE:"):
            result["confidence"] = line.split(":", 1)[1].strip().lower()

    return result


def validate_coaching(athlete_sample, client, ctx=None):
    """
    Validate each coaching claim against the actual video content.

    Args:
        athlete_sample: FiftyOne sample with coaching_feedback and tl_video_id
        client: TwelveLabs client
        ctx: Optional operator context for progress updates

    Returns:
        dict with claims, grounding_score, and hallucinations_flagged
    """
    coaching_text = athlete_sample.get("coaching_feedback", "")
    video_id = athlete_sample.get("tl_video_id", "")

    if not coaching_text or not video_id:
        return {
            "claims": [],
            "grounding_score": 0.0,
            "hallucinations_flagged": 0,
        }

    # Step 1: Extract specific factual claims from coaching feedback
    if ctx:
        ctx.set_progress(label="Extracting coaching claims...", progress=0.1)

    extraction_prompt = f"""From this coaching feedback, extract each specific factual claim
about the athlete's technique. For each claim, provide:
- The exact claim made
- The timestamp referenced (if any)
- What body part or movement it's about

Coaching feedback:
{coaching_text}

Format EXACTLY as:
CLAIM 1: [claim text] | TIMESTAMP: [time] | ABOUT: [body part/movement]
CLAIM 2: [claim text] | TIMESTAMP: [time] | ABOUT: [body part/movement]
"""

    claims_result = client.analyze(
        video_id=video_id,
        prompt=extraction_prompt,
    )
    claims_raw = getattr(claims_result, "data", None) or str(claims_result)
    if not isinstance(claims_raw, str):
        claims_raw = str(claims_raw)

    claims = parse_claims(claims_raw)
    if not claims:
        return {
            "claims": [],
            "grounding_score": 0.0,
            "hallucinations_flagged": 0,
        }

    # Step 2: Independently verify each claim
    validated_claims = []
    for i, claim in enumerate(claims):
        if ctx:
            ctx.set_progress(
                label=f"Verifying claim {i + 1}/{len(claims)}...",
                progress=0.2 + (0.7 * i / len(claims)),
            )

        verification_prompt = f"""Look at this video at approximately {claim['timestamp']}.

Describe exactly what you see regarding {claim['about']}.
Be very specific about position, angle, and movement.

Then answer: Does the following statement accurately describe what's happening?

Statement: "{claim['text']}"

Answer with EXACTLY this format:
ACCURATE: [yes or no]
WHAT I SEE: [your independent observation]
CONFIDENCE: [high, medium, or low]
"""

        try:
            verification_result = client.analyze(
                video_id=video_id,
                prompt=verification_prompt,
            )
            ver_raw = getattr(verification_result, "data", None) or str(verification_result)
            if not isinstance(ver_raw, str):
                ver_raw = str(ver_raw)

            parsed = parse_verification(ver_raw)
            validated_claims.append({
                "original_claim": claim["text"],
                "timestamp": claim["timestamp"],
                "verification_response": parsed["what_i_see"],
                "is_grounded": parsed["accurate"],
                "confidence": parsed["confidence"],
            })
        except Exception as e:
            validated_claims.append({
                "original_claim": claim["text"],
                "timestamp": claim["timestamp"],
                "verification_response": f"Verification failed: {e}",
                "is_grounded": False,
                "confidence": "low",
            })

    grounded_count = sum(1 for c in validated_claims if c["is_grounded"])
    total = len(validated_claims)

    return {
        "claims": validated_claims,
        "grounding_score": round(grounded_count / total * 100, 1) if total > 0 else 0.0,
        "hallucinations_flagged": total - grounded_count,
    }
