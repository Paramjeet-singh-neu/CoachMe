"""
CounterVision — "What Should I Have Done?"

For any moment where the AI identifies bad form, CounterVision automatically
finds a real video segment from the reference library showing the correct
version of that exact movement.

Chain: Pegasus (generate counterfactual) → Marengo (search reference library)
"""

import json
import re


def find_correct_form(athlete_sample, reference_index_id, client, sport="boxing"):
    """
    Given an analyzed athlete sample with coaching_feedback, generate
    counterfactual descriptions and find matching correct-form references.

    Args:
        athlete_sample: FiftyOne sample with coaching_feedback and tl_video_id fields
        reference_index_id: TwelveLabs index ID for the reference video library
        client: TwelveLabs client instance
        sport: Sport name for context in prompts

    Returns:
        list[dict]: List of problem→correction pairs with reference segments
    """
    coaching_text = athlete_sample["coaching_feedback"]
    video_id = athlete_sample["tl_video_id"]

    if not coaching_text or not video_id:
        return []

    # Step 1: Ask Pegasus to generate counterfactual corrections
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
]"""

    try:
        counterfactual_response = client.analyze(
            video_id=video_id,
            prompt=counterfactual_prompt,
            temperature=0.3,
            max_tokens=1500,
        )
    except Exception as e:
        print(f"[CounterVision] Pegasus counterfactual generation failed: {e}")
        return []

    # Step 2: Parse the counterfactual pairs
    pairs = _parse_counterfactual_json(counterfactual_response.data)
    if not pairs:
        print("[CounterVision] Could not parse counterfactual pairs from Pegasus response")
        return []

    # Step 3: For each correction, search reference library with Marengo
    matches = []
    for pair in pairs:
        search_query = f"{sport} correct technique: {pair['correct_form']}"

        try:
            search_results = client.search.query(
                index_id=reference_index_id,
                search_options=["visual"],
                query_text=search_query,
                page_limit=3,
            )

            # Get best match from paginated results
            best_match = None
            for item in search_results:
                best_match = item
                break  # Take the top result

            if best_match:
                matches.append({
                    "problem_description": pair["problem"],
                    "correct_description": pair["correct_form"],
                    "timestamp": pair.get("timestamp", "unknown"),
                    "reference_video_id": best_match.video_id,
                    "reference_start": best_match.start,
                    "reference_end": best_match.end,
                    "confidence": getattr(best_match, "score", None)
                        or getattr(best_match, "rank", 0.0),
                })
            else:
                # No reference found — still include the correction text
                matches.append({
                    "problem_description": pair["problem"],
                    "correct_description": pair["correct_form"],
                    "timestamp": pair.get("timestamp", "unknown"),
                    "reference_video_id": None,
                    "reference_start": None,
                    "reference_end": None,
                    "confidence": 0.0,
                })

        except Exception as e:
            print(f"[CounterVision] Marengo search failed for pair: {e}")
            matches.append({
                "problem_description": pair["problem"],
                "correct_description": pair["correct_form"],
                "timestamp": pair.get("timestamp", "unknown"),
                "reference_video_id": None,
                "reference_start": None,
                "reference_end": None,
                "confidence": 0.0,
            })

    return matches


def _parse_counterfactual_json(raw_text):
    """Parse Pegasus response into list of counterfactual pairs.

    Tries JSON parsing first, falls back to regex extraction.
    """
    if not raw_text:
        return []

    # Try direct JSON parse
    try:
        # Find JSON array in the response (Pegasus may add surrounding text)
        json_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        if json_match:
            pairs = json.loads(json_match.group())
            if isinstance(pairs, list) and len(pairs) > 0:
                # Validate structure
                valid = []
                for p in pairs:
                    if isinstance(p, dict) and "problem" in p and "correct_form" in p:
                        valid.append({
                            "problem": str(p["problem"]),
                            "timestamp": str(p.get("timestamp", "unknown")),
                            "correct_form": str(p["correct_form"]),
                        })
                return valid if valid else None
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: regex-based extraction
    pairs = []
    problem_pattern = r'(?:PROBLEM|problem|Problem)[:\s]*(.+?)(?:\n|$)'
    correct_pattern = r'(?:CORRECT[_ ]FORM|correct[_ ]form|Correct[_ ]Form)[:\s]*(.+?)(?:\n|$)'
    timestamp_pattern = r'(?:TIMESTAMP|timestamp|Timestamp)[:\s]*(.+?)(?:\n|$)'

    problems = re.findall(problem_pattern, raw_text)
    corrects = re.findall(correct_pattern, raw_text)
    timestamps = re.findall(timestamp_pattern, raw_text)

    for i in range(min(len(problems), len(corrects))):
        pairs.append({
            "problem": problems[i].strip(),
            "timestamp": timestamps[i].strip() if i < len(timestamps) else "unknown",
            "correct_form": corrects[i].strip(),
        })

    return pairs if pairs else None
