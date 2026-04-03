"""
TechniqueSync — Temporal Phase Alignment

Aligns an athlete video and a reference video at matching movement phases
so you can compare timing, duration, and form at each stage.

Powered by: Pegasus (phase extraction) + Marengo (semantic matching)
"""

import re


def parse_phases(raw_text):
    """Parse Pegasus phase breakdown into structured list."""
    phases = []

    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Format: PHASE N: [name] | [start]-[end] | [description]
        match = re.match(
            r"PHASE\s*\d*:?\s*(.+?)\s*\|\s*([\d.]+)\s*-\s*([\d.]+)\s*\|\s*(.+)",
            line,
            re.IGNORECASE,
        )
        if match:
            phases.append({
                "name": match.group(1).strip(),
                "start": float(match.group(2)),
                "end": float(match.group(3)),
                "description": match.group(4).strip(),
            })
            continue

        # Fallback: try simpler formats like "1. Name (0.0s - 2.0s): description"
        match2 = re.match(
            r"\d+[.)]\s*(.+?)\s*\(?([\d.]+)s?\s*[-–]\s*([\d.]+)s?\)?\s*:?\s*(.*)",
            line,
        )
        if match2:
            phases.append({
                "name": match2.group(1).strip(),
                "start": float(match2.group(2)),
                "end": float(match2.group(3)),
                "description": match2.group(4).strip() or match2.group(1).strip(),
            })

    return phases


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
    athlete_vid = athlete_sample.get("tl_video_id", "")
    reference_vid = reference_sample.get("tl_video_id", "")

    if not athlete_vid or not reference_vid:
        return {"phases": [], "overall_sync_score": 0.0, "skill": skill_name}

    phase_prompt = f"""Break down this {skill_name} technique video into sequential movement
phases. For each phase provide the phase name, start and end timestamps
in seconds, and a brief description of body position and movement.

Format EXACTLY as:
PHASE 1: [name] | [start_seconds]-[end_seconds] | [description]
PHASE 2: [name] | [start_seconds]-[end_seconds] | [description]
PHASE 3: [name] | [start_seconds]-[end_seconds] | [description]

Example:
PHASE 1: Stance Setup | 0.0-1.2 | Feet shoulder-width apart, hands up in guard position
PHASE 2: Weight Shift | 1.2-2.0 | Weight transfers to back foot, hip begins rotation
"""

    # Step 1: Extract phases from both videos
    if ctx:
        ctx.set_progress(label="Analyzing athlete movement phases...", progress=0.1)

    athlete_phases_result = client.analyze(
        video_id=athlete_vid,
        prompt=phase_prompt,
    )
    athlete_raw = getattr(athlete_phases_result, "data", None) or str(athlete_phases_result)
    if not isinstance(athlete_raw, str):
        athlete_raw = str(athlete_raw)

    if ctx:
        ctx.set_progress(label="Analyzing reference movement phases...", progress=0.3)

    ref_phases_result = client.analyze(
        video_id=reference_vid,
        prompt=phase_prompt,
    )
    ref_raw = getattr(ref_phases_result, "data", None) or str(ref_phases_result)
    if not isinstance(ref_raw, str):
        ref_raw = str(ref_raw)

    athlete_phases = parse_phases(athlete_raw)
    reference_phases = parse_phases(ref_raw)

    if not athlete_phases or not reference_phases:
        return {"phases": [], "overall_sync_score": 0.0, "skill": skill_name}

    # Step 2: Match phases by name similarity (simple keyword overlap)
    if ctx:
        ctx.set_progress(label="Aligning movement phases...", progress=0.6)

    alignments = []
    used_ref_phases = set()

    for a_phase in athlete_phases:
        best_match = None
        best_score = -1

        for j, r_phase in enumerate(reference_phases):
            if j in used_ref_phases:
                continue
            # Simple word overlap similarity
            a_words = set(a_phase["name"].lower().split() + a_phase["description"].lower().split())
            r_words = set(r_phase["name"].lower().split() + r_phase["description"].lower().split())
            overlap = len(a_words & r_words)
            total = len(a_words | r_words)
            score = overlap / total if total > 0 else 0

            if score > best_score:
                best_score = score
                best_match = (j, r_phase)

        if best_match is None:
            # No match — pair with next unused reference phase
            for j, r_phase in enumerate(reference_phases):
                if j not in used_ref_phases:
                    best_match = (j, r_phase)
                    best_score = 0.1
                    break

        if best_match:
            used_ref_phases.add(best_match[0])
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

    # Compute overall sync score
    if alignments:
        overall_sync = round(
            sum(a["alignment_score"] for a in alignments) / len(alignments), 3
        )
    else:
        overall_sync = 0.0

    if ctx:
        ctx.set_progress(label="Phase alignment complete!", progress=1.0)

    return {
        "phases": alignments,
        "overall_sync_score": overall_sync,
        "skill": skill_name,
    }
