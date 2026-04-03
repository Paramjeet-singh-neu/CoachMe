"""
CounterVision — "What Should I Have Done?"

For each problem identified in coaching feedback, auto-finds a real video
segment from the reference library showing the correct form for that exact
movement.

Powered by: Pegasus (counterfactual generation) + Marengo (semantic search)
"""

import json
import re


def _parse_counterfactual_json(raw_text):
    """Parse Pegasus response into list of counterfactual pairs.

    Tries JSON parsing first, falls back to regex extraction.
    """
    if not raw_text:
        return []

    # Tier 1: JSON extraction (most reliable when Pegasus cooperates)
    try:
        json_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        if json_match:
            pairs = json.loads(json_match.group())
            if isinstance(pairs, list) and len(pairs) > 0:
                valid = []
                for p in pairs:
                    if isinstance(p, dict) and "problem" in p and "correct_form" in p:
                        valid.append({
                            "problem": str(p["problem"]),
                            "timestamp": str(p.get("timestamp", "0:00")),
                            "correct_form": str(p["correct_form"]),
                        })
                if valid:
                    return valid
    except (json.JSONDecodeError, AttributeError):
        pass

    # Tier 2: Regex fallback for labeled text format
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

    return pairs if pairs else []


def find_correct_form(athlete_sample, reference_index_id, client, ctx=None, sport=None):
    """
    For each coaching problem, find the matching correct-form reference segment.

    Args:
        athlete_sample: FiftyOne sample with coaching_feedback and tl_video_id
        reference_index_id: Twelve Labs index ID containing reference videos
        client: TwelveLabs client
        ctx: Optional operator context for progress updates
        sport: Optional sport override (falls back to sample field)

    Returns:
        list[dict] — counterfactual matches with reference segments
    """
    coaching_text = athlete_sample.get("coaching_feedback", "")
    if sport is None:
        sport = athlete_sample.get("sport", "general")
    video_id = athlete_sample.get("tl_video_id", "")

    if not coaching_text or not video_id:
        return []

    # Step 1: Ask Pegasus to generate counterfactual descriptions
    if ctx:
        ctx.set_progress(label="Generating counterfactual descriptions...", progress=0.2)

    counterfactual_prompt = f"""You are analyzing coaching feedback for a {sport} athlete.

Based on this coaching feedback:

{coaching_text}

For each specific technique problem identified, generate a JSON array where each element has:
- "problem": brief description of the error (1 sentence)
- "timestamp": approximate time in the video (e.g., "0:04-0:06")
- "correct_form": very specific description of what CORRECT technique looks like for that movement — include body positioning, angles, timing, and mechanics (1-2 sentences)

Return ONLY valid JSON array, no other text. Example:
[
  {{"problem": "Left elbow drops before jab extension", "timestamp": "0:04-0:06", "correct_form": "Elbow stays tucked close to ribcage, arm extends in a straight line from shoulder with fist rotating at full extension"}}
]

If JSON is not possible, format as:
PROBLEM: [description]
CORRECT FORM: [description]
TIMESTAMP: [time]
"""

    try:
        counterfactual_result = client.analyze(
            video_id=video_id,
            prompt=counterfactual_prompt,
            temperature=0.3,
            max_tokens=1500,
        )
        raw_text = getattr(counterfactual_result, "data", None) or str(counterfactual_result)
        if not isinstance(raw_text, str):
            raw_text = str(raw_text)
    except Exception as e:
        print(f"[CounterVision] Pegasus counterfactual generation failed: {e}")
        return []

    # Step 2: Parse the counterfactual pairs (JSON first, regex fallback)
    pairs = _parse_counterfactual_json(raw_text)
    if not pairs:
        print("[CounterVision] Could not parse counterfactual pairs from Pegasus response")
        return []

    # Step 3: For each correction, search reference library with Marengo
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
                start = getattr(top, "start", 0)
                end = getattr(top, "end", 0)
                score = getattr(top, "score", None)
                rank = getattr(top, "rank", 1) or 1
                # Use score if available, otherwise derive from rank
                confidence = round(float(score) * 100, 1) if score else round(max(0, 100 - (rank - 1) * 15), 1)

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
            print(f"[CounterVision] Marengo search failed for pair: {e}")
            matches.append({
                "problem_description": pair["problem"],
                "correct_description": pair["correct_form"],
                "timestamp": pair["timestamp"],
                "reference_video_id": "",
                "reference_start": 0.0,
                "reference_end": 0.0,
                "confidence": 0.0,
            })

    return matches
