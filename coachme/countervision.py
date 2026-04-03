"""
CounterVision — "What Should I Have Done?"

For each problem identified in coaching feedback, auto-finds a real video
segment from the reference library showing the correct form for that exact
movement.

Powered by: Pegasus (counterfactual generation) + Marengo (semantic search)
"""

import re


def parse_counterfactual_pairs(raw_text):
    """Parse Pegasus output into structured problem/correction pairs."""
    pairs = []
    current = {}

    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.upper().startswith("PROBLEM:"):
            if current.get("problem"):
                pairs.append(current)
            current = {"problem": line.split(":", 1)[1].strip()}
        elif line.upper().startswith("CORRECT FORM:") or line.upper().startswith("CORRECT:"):
            current["correct_form"] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("TIMESTAMP:"):
            current["timestamp"] = line.split(":", 1)[1].strip()

    if current.get("problem"):
        pairs.append(current)

    # Ensure all pairs have required fields
    for p in pairs:
        p.setdefault("correct_form", "Proper technique with correct body positioning")
        p.setdefault("timestamp", "0:00")

    return pairs


def find_correct_form(athlete_sample, reference_index_id, client, ctx=None):
    """
    For each coaching problem, find the matching correct-form reference segment.

    Args:
        athlete_sample: FiftyOne sample with coaching_feedback and tl_video_id
        reference_index_id: Twelve Labs index ID containing reference videos
        client: TwelveLabs client
        ctx: Optional operator context for progress updates

    Returns:
        list[dict] — counterfactual matches with reference segments
    """
    coaching_text = athlete_sample.get("coaching_feedback", "")
    sport = athlete_sample.get("sport", "general")
    video_id = athlete_sample.get("tl_video_id", "")

    if not coaching_text or not video_id:
        return []

    # Step 1: Ask Pegasus to generate counterfactual descriptions
    if ctx:
        ctx.set_progress(label="Generating counterfactual descriptions...", progress=0.2)

    counterfactual_prompt = f"""Based on this coaching feedback for a {sport} athlete:

{coaching_text}

For each specific problem identified, describe what CORRECT technique
looks like for that exact moment. Be very specific about body positioning,
timing, and movement mechanics.

Format EXACTLY as (one group per problem):
PROBLEM: [brief description of the error]
CORRECT FORM: [specific description of what correct form looks like]
TIMESTAMP: [when in the video this occurs, e.g. 0:04]
"""

    counterfactual_result = client.analyze(
        video_id=video_id,
        prompt=counterfactual_prompt,
    )
    raw_text = getattr(counterfactual_result, "data", None) or str(counterfactual_result)
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)

    pairs = parse_counterfactual_pairs(raw_text)
    if not pairs:
        return []

    # Step 2: For each correction, search reference library with Marengo
    matches = []
    for i, pair in enumerate(pairs):
        if ctx:
            ctx.set_progress(
                label=f"Searching reference library ({i + 1}/{len(pairs)})...",
                progress=0.4 + (0.5 * i / len(pairs)),
            )

        search_query = f"{sport} correct technique: {pair['correct_form']}"

        try:
            search_results = client.search.query(
                index_id=reference_index_id,
                query_text=search_query,
                search_options=["visual"],
            )

            search_data = list(search_results)
            if search_data:
                top = search_data[0]
                vid_id = getattr(top, "video_id", None) or getattr(top, "id", None)
                rank = getattr(top, "rank", 1) or 1
                start = getattr(top, "start", 0)
                end = getattr(top, "end", 0)
                # Top result gets high confidence, decreasing by rank
                confidence = round(max(0, 100 - (rank - 1) * 15), 1)

                matches.append({
                    "problem_description": pair["problem"],
                    "correct_description": pair["correct_form"],
                    "timestamp": pair["timestamp"],
                    "reference_video_id": vid_id or "",
                    "reference_start": float(start) if start else 0.0,
                    "reference_end": float(end) if end else 0.0,
                    "confidence": confidence,
                })
            else:
                matches.append({
                    "problem_description": pair["problem"],
                    "correct_description": pair["correct_form"],
                    "timestamp": pair["timestamp"],
                    "reference_video_id": "",
                    "reference_start": 0.0,
                    "reference_end": 0.0,
                    "confidence": 0.0,
                })
        except Exception as e:
            matches.append({
                "problem_description": pair["problem"],
                "correct_description": pair["correct_form"],
                "timestamp": pair["timestamp"],
                "reference_video_id": "",
                "reference_start": 0.0,
                "reference_end": 0.0,
                "confidence": 0.0,
                "error": str(e),
            })

    return matches
