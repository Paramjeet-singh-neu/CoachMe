"""
TechniqueSync — Temporal Phase Alignment

Aligns an athlete video and a reference video at matching movement phases
so you can compare timing, duration, and form at each stage.

Powered by: Pegasus (phase extraction) + Marengo (semantic embedding matching)
"""

import json
import re
import numpy as np


def _safe_get(sample, field, default=None):
    """Safely get a field from a FiftyOne sample."""
    try:
        val = sample[field]
        return val if val is not None else default
    except (KeyError, AttributeError):
        return default


def _parse_phases(raw_text):
    """Parse Pegasus phase response into structured list.

    Tries JSON first, falls back to regex.
    """
    if not raw_text:
        return []

    # Tier 1: JSON extraction
    try:
        json_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        if json_match:
            phases = json.loads(json_match.group())
            if isinstance(phases, list) and len(phases) > 0:
                valid = []
                for p in phases:
                    if isinstance(p, dict) and ("phase" in p or "name" in p):
                        valid.append({
                            "name": str(p.get("phase") or p.get("name", "")),
                            "start": float(p.get("start", 0)),
                            "end": float(p.get("end", 0)),
                            "description": str(p.get("description", p.get("phase") or p.get("name", ""))),
                        })
                if valid:
                    return valid
    except (json.JSONDecodeError, ValueError, AttributeError):
        pass

    # Tier 2: Regex for "PHASE N: name | start-end | description"
    phases = []
    pattern = re.compile(
        r"PHASE\s*\d*:?\s*(.+?)\s*\|\s*([\d.]+)\s*-\s*([\d.]+)\s*\|\s*(.+)",
        re.IGNORECASE,
    )
    for line in raw_text.strip().split("\n"):
        match = pattern.match(line.strip())
        if match:
            phases.append({
                "name": match.group(1).strip(),
                "start": float(match.group(2)),
                "end": float(match.group(3)),
                "description": match.group(4).strip(),
            })
            continue

        # Tier 3: "1. Name (0.0s - 2.0s): description"
        match2 = re.match(
            r"\d+[.)]\s*(.+?)\s*\(?([\d.]+)s?\s*[-\u2013]\s*([\d.]+)s?\)?\s*:?\s*(.*)",
            line.strip(),
        )
        if match2:
            phases.append({
                "name": match2.group(1).strip(),
                "start": float(match2.group(2)),
                "end": float(match2.group(3)),
                "description": match2.group(4).strip() or match2.group(1).strip(),
            })

    return phases


def _get_text_embeddings(texts, client):
    """Get Marengo text embeddings for a list of strings.

    Returns list of embedding vectors, or None if the API call fails.
    """
    try:
        from twelvelabs.types import TextInputRequest
        embeddings = []
        for text in texts:
            response = client.embed.v_2.create(
                input_type="text",
                model_name="marengo3.0",
                text=TextInputRequest(input_text=text),
            )
            if response.data and len(response.data) > 0:
                embeddings.append(response.data[0].embedding)
            else:
                return None
        return embeddings
    except Exception as e:
        print(f"[TechniqueSync] Text embedding failed: {e}")
        try:
            embeddings = []
            for text in texts:
                response = client.embed.v_2.create(
                    input_type="text",
                    model_name="marengo3.0",
                    text={"input_text": text},
                )
                if response.data and len(response.data) > 0:
                    embeddings.append(response.data[0].embedding)
                else:
                    return None
            return embeddings
        except Exception as e2:
            print(f"[TechniqueSync] Text embedding fallback also failed: {e2}")
            return None


def _align_phases_by_embedding(athlete_phases, reference_phases, client):
    """Match phases using Marengo text embeddings + cosine similarity.

    Falls back to word-overlap matching if embeddings fail.
    """
    # Get embeddings for all phase descriptions
    athlete_embeddings = _get_text_embeddings(
        [p["description"] for p in athlete_phases], client
    )
    reference_embeddings = _get_text_embeddings(
        [p["description"] for p in reference_phases], client
    )

    # If embedding API fails, fall back to word-overlap matching
    if athlete_embeddings is None or reference_embeddings is None:
        return _align_phases_by_words(athlete_phases, reference_phases)

    # Compute cosine similarity matrix
    a_emb = np.array(athlete_embeddings)
    r_emb = np.array(reference_embeddings)

    a_norm = a_emb / (np.linalg.norm(a_emb, axis=1, keepdims=True) + 1e-10)
    r_norm = r_emb / (np.linalg.norm(r_emb, axis=1, keepdims=True) + 1e-10)

    similarity_matrix = a_norm @ r_norm.T  # (N, M)

    # Greedy alignment: for each athlete phase, find best reference match
    used_refs = set()
    alignments = []

    for i, a_phase in enumerate(athlete_phases):
        best_j = -1
        best_score = -1.0

        for j in range(len(reference_phases)):
            if j not in used_refs and similarity_matrix[i, j] > best_score:
                best_score = similarity_matrix[i, j]
                best_j = j

        if best_j >= 0:
            used_refs.add(best_j)
            r_phase = reference_phases[best_j]

            a_duration = a_phase["end"] - a_phase["start"]
            r_duration = r_phase["end"] - r_phase["start"]

            alignments.append({
                "athlete_phase": a_phase["name"],
                "athlete_time": f"{a_phase['start']:.1f}s-{a_phase['end']:.1f}s",
                "athlete_description": a_phase["description"],
                "reference_phase": r_phase["name"],
                "reference_time": f"{r_phase['start']:.1f}s-{r_phase['end']:.1f}s",
                "reference_description": r_phase["description"],
                "timing_delta_seconds": round(a_duration - r_duration, 2),
                "alignment_score": round(float(best_score), 3),
            })

    return alignments


def _align_phases_by_words(athlete_phases, reference_phases):
    """Fallback: align phases by word-overlap similarity."""
    used_refs = set()
    alignments = []

    for a_phase in athlete_phases:
        best_match = None
        best_score = -1

        for j, r_phase in enumerate(reference_phases):
            if j in used_refs:
                continue
            a_words = set(a_phase["name"].lower().split() + a_phase["description"].lower().split())
            r_words = set(r_phase["name"].lower().split() + r_phase["description"].lower().split())
            overlap = len(a_words & r_words)
            total = len(a_words | r_words)
            score = overlap / total if total > 0 else 0

            if score > best_score:
                best_score = score
                best_match = (j, r_phase)

        if best_match is None:
            for j, r_phase in enumerate(reference_phases):
                if j not in used_refs:
                    best_match = (j, r_phase)
                    best_score = 0.1
                    break

        if best_match:
            used_refs.add(best_match[0])
            r_phase = best_match[1]
            a_duration = a_phase["end"] - a_phase["start"]
            r_duration = r_phase["end"] - r_phase["start"]

            alignments.append({
                "athlete_phase": a_phase["name"],
                "athlete_time": f"{a_phase['start']:.1f}s-{a_phase['end']:.1f}s",
                "athlete_description": a_phase["description"],
                "reference_phase": r_phase["name"],
                "reference_time": f"{r_phase['start']:.1f}s-{r_phase['end']:.1f}s",
                "reference_description": r_phase["description"],
                "timing_delta_seconds": round(a_duration - r_duration, 2),
                "alignment_score": round(best_score, 3),
            })

    return alignments


def sync_technique(athlete_sample, reference_sample, skill_name, client, ctx=None):
    """
    Align movement phases between athlete and reference videos.

    Args:
        athlete_sample: FiftyOne sample (athlete) with tl_video_id
        reference_sample: FiftyOne sample (reference) with tl_video_id
        skill_name: Name of the skill/movement (e.g. "jab", "squat")
        client: TwelveLabs client
        ctx: Optional operator context for progress updates

    Returns:
        dict with phases, overall_sync_score, and skill name
    """
    athlete_vid = _safe_get(athlete_sample, "tl_video_id", "")
    reference_vid = _safe_get(reference_sample, "tl_video_id", "")

    if not athlete_vid or not reference_vid:
        return {"phases": [], "overall_sync_score": 0.0, "skill": skill_name}

    phase_prompt = f"""Break down this {skill_name} technique video into sequential movement phases.

For each phase, provide:
- Phase name (e.g., "stance setup", "backswing", "contact", "follow-through")
- Start time in seconds
- End time in seconds
- Brief description of body position and movement

Return ONLY valid JSON array, no other text. Example:
[
  {{"phase": "stance setup", "start": 0.0, "end": 1.5, "description": "Feet shoulder-width apart, knees slightly bent, hands up in guard position"}},
  {{"phase": "weight shift", "start": 1.5, "end": 2.8, "description": "Weight transfers from back foot to front foot, hips begin rotating"}}
]

If JSON is not possible, format as:
PHASE 1: [name] | [start_seconds]-[end_seconds] | [description]
PHASE 2: [name] | [start_seconds]-[end_seconds] | [description]
"""

    # Step 1: Extract phases from both videos
    if ctx:
        ctx.set_progress(label="Analyzing athlete movement phases...", progress=0.1)

    try:
        athlete_phases_result = client.analyze(
            video_id=athlete_vid,
            prompt=phase_prompt,
            temperature=0.2,
            max_tokens=1500,
        )
        athlete_raw = getattr(athlete_phases_result, "data", None) or str(athlete_phases_result)
        if not isinstance(athlete_raw, str):
            athlete_raw = str(athlete_raw)
    except Exception as e:
        return {"phases": [], "overall_sync_score": 0.0, "skill": skill_name, "error": f"Athlete analysis failed: {e}"}

    if ctx:
        ctx.set_progress(label="Analyzing reference movement phases...", progress=0.3)

    try:
        ref_phases_result = client.analyze(
            video_id=reference_vid,
            prompt=phase_prompt,
            temperature=0.2,
            max_tokens=1500,
        )
        ref_raw = getattr(ref_phases_result, "data", None) or str(ref_phases_result)
        if not isinstance(ref_raw, str):
            ref_raw = str(ref_raw)
    except Exception as e:
        return {"phases": [], "overall_sync_score": 0.0, "skill": skill_name, "error": f"Reference analysis failed: {e}"}

    # Step 2: Parse phases
    athlete_phases = _parse_phases(athlete_raw)
    reference_phases = _parse_phases(ref_raw)

    if not athlete_phases or not reference_phases:
        return {"phases": [], "overall_sync_score": 0.0, "skill": skill_name}

    # Step 3: Match phases using Marengo embeddings (with word-overlap fallback)
    if ctx:
        ctx.set_progress(label="Aligning movement phases with Marengo embeddings...", progress=0.6)

    alignments = _align_phases_by_embedding(athlete_phases, reference_phases, client)

    # Step 4: Compute overall sync score
    if alignments:
        overall_sync = round(
            float(np.mean([a["alignment_score"] for a in alignments])), 3
        )
    else:
        overall_sync = 0.0

    if ctx:
        ctx.set_progress(label="Phase alignment complete!", progress=1.0)

    return {
        "phases": alignments,
        "overall_sync_score": overall_sync,
        "skill": skill_name,
        "athlete_phase_count": len(athlete_phases),
        "reference_phase_count": len(reference_phases),
    }
