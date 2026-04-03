"""
TechniqueSync — Temporal Phase Alignment

Given an athlete video and a reference pro video of the same skill,
TechniqueSync aligns them at matching movement phases so you can
compare side-by-side at each stage.

Chain: Pegasus (extract phases from both) → Marengo embeddings (match phases)
"""

import json
import re
import numpy as np


def sync_technique(athlete_sample, reference_sample, skill_name, client):
    """
    Temporally align an athlete video against a reference video by
    matching movement phases.

    Args:
        athlete_sample: FiftyOne sample with tl_video_id
        reference_sample: FiftyOne sample with tl_video_id
        skill_name: Name of the skill (e.g., "jab", "squat", "serve")
        client: TwelveLabs client instance

    Returns:
        dict with phase alignments, timing deltas, and overall sync score
    """
    athlete_video_id = athlete_sample["tl_video_id"]
    reference_video_id = reference_sample["tl_video_id"]

    if not athlete_video_id or not reference_video_id:
        return {"error": "Missing tl_video_id on one or both samples"}

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
]"""

    # Step 1: Extract phases from both videos in sequence
    # (Can't parallelize — TwelveLabs client is synchronous)
    try:
        athlete_phases_raw = client.analyze(
            video_id=athlete_video_id,
            prompt=phase_prompt,
            temperature=0.2,
            max_tokens=1500,
        )
    except Exception as e:
        return {"error": f"Failed to analyze athlete video: {e}"}

    try:
        reference_phases_raw = client.analyze(
            video_id=reference_video_id,
            prompt=phase_prompt,
            temperature=0.2,
            max_tokens=1500,
        )
    except Exception as e:
        return {"error": f"Failed to analyze reference video: {e}"}

    # Step 2: Parse phase lists
    athlete_phases = _parse_phases(athlete_phases_raw.data)
    reference_phases = _parse_phases(reference_phases_raw.data)

    if not athlete_phases:
        return {"error": "Could not parse athlete video phases"}
    if not reference_phases:
        return {"error": "Could not parse reference video phases"}

    # Step 3: Match phases using text embedding similarity
    # We embed each phase description and match by cosine similarity
    alignments = _align_phases(athlete_phases, reference_phases, client)

    # Step 4: Compute overall sync score
    if alignments:
        scores = [a["alignment_score"] for a in alignments]
        overall_sync = float(np.mean(scores))
    else:
        overall_sync = 0.0

    return {
        "phases": alignments,
        "overall_sync_score": round(overall_sync, 3),
        "skill": skill_name,
        "athlete_phase_count": len(athlete_phases),
        "reference_phase_count": len(reference_phases),
    }


def _align_phases(athlete_phases, reference_phases, client):
    """Match athlete phases to reference phases using text similarity.

    Uses Marengo text embeddings to compute cosine similarity between
    phase descriptions, then does greedy best-match alignment.
    """
    # Get embeddings for all phase descriptions
    athlete_embeddings = _get_text_embeddings(
        [p["description"] for p in athlete_phases], client
    )
    reference_embeddings = _get_text_embeddings(
        [p["description"] for p in reference_phases], client
    )

    # If embedding API fails, fall back to name-based matching
    if athlete_embeddings is None or reference_embeddings is None:
        return _align_phases_by_name(athlete_phases, reference_phases)

    # Compute similarity matrix
    # athlete_embeddings: (N, D), reference_embeddings: (M, D)
    a_emb = np.array(athlete_embeddings)
    r_emb = np.array(reference_embeddings)

    # Normalize for cosine similarity
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
                "athlete_phase": a_phase["phase"],
                "athlete_time": f"{a_phase['start']:.1f}s-{a_phase['end']:.1f}s",
                "athlete_description": a_phase["description"],
                "reference_phase": r_phase["phase"],
                "reference_time": f"{r_phase['start']:.1f}s-{r_phase['end']:.1f}s",
                "reference_description": r_phase["description"],
                "timing_delta_seconds": round(a_duration - r_duration, 2),
                "alignment_score": round(float(best_score), 3),
            })

    return alignments


def _align_phases_by_name(athlete_phases, reference_phases):
    """Fallback: align phases by sequential order if embeddings unavailable."""
    alignments = []
    n = min(len(athlete_phases), len(reference_phases))

    for i in range(n):
        a = athlete_phases[i]
        r = reference_phases[i]

        a_duration = a["end"] - a["start"]
        r_duration = r["end"] - r["start"]

        alignments.append({
            "athlete_phase": a["phase"],
            "athlete_time": f"{a['start']:.1f}s-{a['end']:.1f}s",
            "athlete_description": a["description"],
            "reference_phase": r["phase"],
            "reference_time": f"{r['start']:.1f}s-{r['end']:.1f}s",
            "reference_description": r["description"],
            "timing_delta_seconds": round(a_duration - r_duration, 2),
            "alignment_score": 0.5,  # Unknown — no embedding comparison
        })

    return alignments


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
        # Try alternative import path
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


def _parse_phases(raw_text):
    """Parse Pegasus phase response into structured list.

    Tries JSON first, falls back to regex.
    """
    if not raw_text:
        return []

    # Try JSON parse
    try:
        json_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        if json_match:
            phases = json.loads(json_match.group())
            if isinstance(phases, list) and len(phases) > 0:
                valid = []
                for p in phases:
                    if isinstance(p, dict) and "phase" in p:
                        valid.append({
                            "phase": str(p["phase"]),
                            "start": float(p.get("start", 0)),
                            "end": float(p.get("end", 0)),
                            "description": str(p.get("description", p["phase"])),
                        })
                return valid if valid else None
    except (json.JSONDecodeError, ValueError, AttributeError):
        pass

    # Fallback: regex for "PHASE N: name | start-end | description"
    phase_pattern = r'PHASE\s*\d+[:\s]*([^|]+)\|\s*([\d.]+)\s*-\s*([\d.]+)\s*\|\s*(.+?)(?:\n|$)'
    found = re.findall(phase_pattern, raw_text, re.IGNORECASE)

    if found:
        return [
            {
                "phase": name.strip(),
                "start": float(start),
                "end": float(end),
                "description": desc.strip(),
            }
            for name, start, end, desc in found
        ]

    return []
