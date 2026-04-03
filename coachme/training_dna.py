"""
TrainingDNA — Reference Library Health Card

Auto-generates a DNA profile of your reference video collection:
what techniques are covered, what's missing, what angles are overrepresented,
and whether there are near-duplicate clips.

Powered by: Pegasus (video description) + Marengo (embeddings + similarity)
"""

import re
from collections import Counter


# Common techniques per sport for gap detection
SPORT_TECHNIQUES = {
    "boxing": ["jab", "cross", "hook", "uppercut", "slip", "bob and weave", "footwork", "combination"],
    "tennis": ["serve", "forehand", "backhand", "volley", "overhead", "slice", "drop shot", "footwork"],
    "weightlifting": ["squat", "deadlift", "bench press", "overhead press", "clean", "snatch", "row"],
    "cricket": ["bowling", "batting", "fielding", "cover drive", "pull shot", "forward defense"],
    "yoga": ["downward dog", "warrior", "tree pose", "plank", "cobra", "child pose", "sun salutation"],
    "swimming": ["freestyle", "backstroke", "breaststroke", "butterfly", "flip turn", "dive start"],
}

CAMERA_ANGLES = ["front", "side", "rear", "overhead", "diagonal", "dynamic"]
SKILL_LEVELS = ["beginner", "intermediate", "professional"]


def parse_video_description(raw_text):
    """Parse Pegasus video description into structured dict."""
    result = {
        "technique": "unknown",
        "angle": "unknown",
        "skill_level": "unknown",
        "phases_visible": "",
        "duration_quality": "",
        "summary": "",
    }

    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("TECHNIQUE:"):
            result["technique"] = line.split(":", 1)[1].strip().lower()
        elif line.upper().startswith("ANGLE:"):
            result["angle"] = line.split(":", 1)[1].strip().lower()
        elif line.upper().startswith("SKILL LEVEL:") or line.upper().startswith("SKILL_LEVEL:"):
            result["skill_level"] = line.split(":", 1)[1].strip().lower()
        elif line.upper().startswith("PHASES VISIBLE:") or line.upper().startswith("PHASES:"):
            result["phases_visible"] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("DURATION QUALITY:") or line.upper().startswith("DURATION:"):
            result["duration_quality"] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("SUMMARY:"):
            result["summary"] = line.split(":", 1)[1].strip()

    return result


def profile_references(dataset, sport, client, similarity_threshold=0.92, ctx=None):
    """
    Generate a health card for the reference video library.

    Args:
        dataset: FiftyOne dataset of reference videos
        sport: Sport name
        client: TwelveLabs client
        similarity_threshold: Threshold for flagging near-duplicates (0-1)
        ctx: Optional operator context for progress updates

    Returns:
        dict — the full TrainingDNA profile
    """
    samples = list(dataset)
    total = len(samples)

    if total == 0:
        return {
            "total_videos": 0,
            "technique_distribution": {},
            "angle_distribution": {},
            "skill_level_distribution": {},
            "near_duplicates": [],
            "coverage_gaps": ["No videos in reference library"],
            "recommendations": ["Add reference videos to get started"],
        }

    # Step 1: Describe each video with Pegasus
    descriptions = []
    for i, sample in enumerate(samples):
        if ctx:
            ctx.set_progress(
                label=f"Analyzing video {i + 1}/{total}...",
                progress=(i / total) * 0.6,
            )

        video_id = sample.get("tl_video_id", "")
        if not video_id:
            descriptions.append({
                "technique": "unknown",
                "angle": "unknown",
                "skill_level": "unknown",
                "phases_visible": "",
                "duration_quality": "",
                "summary": "",
                "sample_id": sample.id,
                "filepath": sample.filepath,
            })
            continue

        desc_prompt = f"""Analyze this {sport} technique reference video. Provide:
TECHNIQUE: [specific technique shown, e.g., jab, cross, hook, squat]
ANGLE: [camera angle: front, side, overhead, rear, diagonal, dynamic]
SKILL LEVEL: [beginner, intermediate, professional]
PHASES VISIBLE: [list the movement phases you can see]
DURATION QUALITY: [is the clip long enough to show full technique? yes/no]
SUMMARY: [one sentence description]
"""

        try:
            desc_result = client.analyze(
                video_id=video_id,
                prompt=desc_prompt,
            )
            raw = getattr(desc_result, "data", None) or str(desc_result)
            if not isinstance(raw, str):
                raw = str(raw)
            parsed = parse_video_description(raw)
        except Exception:
            parsed = {
                "technique": "unknown",
                "angle": "unknown",
                "skill_level": "unknown",
                "phases_visible": "",
                "duration_quality": "",
                "summary": "",
            }

        parsed["sample_id"] = sample.id
        parsed["filepath"] = sample.filepath
        descriptions.append(parsed)

    # Step 2: Compute pairwise similarity for near-duplicate detection
    if ctx:
        ctx.set_progress(label="Checking for near-duplicates...", progress=0.7)

    near_dupes = []
    video_ids = [s.get("tl_video_id", "") for s in samples]

    for i in range(len(video_ids)):
        if not video_ids[i]:
            continue
        for j in range(i + 1, len(video_ids)):
            if not video_ids[j]:
                continue
            try:
                # Use text description search to find similar videos
                desc_i = descriptions[i].get("summary", "") or descriptions[i].get("technique", "")
                results = client.search.query(
                    index_id=samples[i].get("tl_index_id", ""),
                    query_text=f"{sport} {desc_i}",
                    search_options=["visual"],
                    page_limit=total,
                )
                search_data = list(results)
                for r in search_data:
                    r_vid = getattr(r, "video_id", None) or getattr(r, "id", None)
                    r_rank = getattr(r, "rank", None)
                    if r_vid == video_ids[j] and r_rank and r_rank <= 2:
                        # Very high rank match suggests near-duplicate
                        sim = round(max(0, (1 - (r_rank - 1) / max(total, 1))) * 100, 1)
                        if sim / 100 > similarity_threshold:
                            near_dupes.append({
                                "video_a": descriptions[i]["filepath"],
                                "video_b": descriptions[j]["filepath"],
                                "similarity": sim,
                            })
                break  # Only need one search per pair i
            except Exception:
                continue

    # Step 3: Analyze distributions
    if ctx:
        ctx.set_progress(label="Analyzing distributions...", progress=0.85)

    techniques = Counter(d["technique"] for d in descriptions if d["technique"] != "unknown")
    angles = Counter(d["angle"] for d in descriptions if d["angle"] != "unknown")
    skill_levels = Counter(d["skill_level"] for d in descriptions if d["skill_level"] != "unknown")

    # Step 4: Identify coverage gaps
    coverage_gaps = []
    sport_lower = sport.lower()
    known_techniques = SPORT_TECHNIQUES.get(sport_lower, [])

    for tech in known_techniques:
        found = any(tech.lower() in t.lower() for t in techniques.keys())
        if not found:
            coverage_gaps.append(f"Missing technique: {tech}")

    unique_angles = set(a.lower() for a in angles.keys())
    if len(unique_angles) <= 1 and total > 1:
        angle_name = list(unique_angles)[0] if unique_angles else "unknown"
        coverage_gaps.append(
            f"Only one camera angle ({angle_name}). Add alternative angles for better analysis."
        )

    if not any("professional" in s.lower() or "pro" in s.lower() for s in skill_levels.keys()):
        if total > 0:
            coverage_gaps.append(
                "No professional-level references detected. Add pro footage for a stronger coaching baseline."
            )

    # Step 5: Generate recommendations
    recommendations = []
    if near_dupes:
        recommendations.append(
            f"{len(near_dupes)} near-duplicate pair(s) found. Consider removing redundant clips to diversify your library."
        )
    if len(techniques) < 3 and total >= 3:
        recommendations.append(
            "Low technique variety. Add videos of different moves/skills for broader coaching coverage."
        )
    for gap in coverage_gaps[:3]:  # Top 3 gaps as recommendations
        if "Missing technique" in gap:
            tech_name = gap.replace("Missing technique: ", "")
            recommendations.append(f"Add reference footage for {tech_name}.")
    if not recommendations:
        recommendations.append("Reference library looks solid! Good technique coverage.")

    if ctx:
        ctx.set_progress(label="Profile complete!", progress=1.0)

    return {
        "total_videos": total,
        "technique_distribution": dict(techniques),
        "angle_distribution": dict(angles),
        "skill_level_distribution": dict(skill_levels),
        "near_duplicates": near_dupes,
        "coverage_gaps": coverage_gaps,
        "recommendations": recommendations,
    }
