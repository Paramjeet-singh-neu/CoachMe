"""
TrainingDNA — Reference Library Health Card

Auto-generates a "DNA profile" of a reference video collection:
what techniques are covered, what's missing, what angles are
overrepresented, and whether there are near-duplicate clips.

Uses: Twelve Labs Pegasus 1.2 (via client.analyze()) +
      Twelve Labs Marengo 3.0 (via client.search.query())
"""

import re
import time
from collections import Counter


def _analyze_with_retry(client, prompt, video_id, max_retries=5):
    """Call client.analyze() with automatic retry on rate-limit (429)."""
    for attempt in range(max_retries):
        try:
            return client.analyze(prompt=prompt, video_id=video_id)
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "too_many_requests" in err_str or "rate" in err_str:
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


# Known technique lists per sport (expandable)
SPORT_TECHNIQUES = {
    "boxing": [
        "jab", "cross", "hook", "uppercut",
        "slip", "bob and weave", "parry", "block",
        "footwork", "stance", "combination",
    ],
    "tennis": [
        "forehand", "backhand", "serve", "volley",
        "lob", "drop shot", "overhead", "slice",
        "footwork", "return",
    ],
    "weightlifting": [
        "squat", "deadlift", "bench press", "overhead press",
        "clean", "snatch", "clean and jerk",
        "front squat", "row",
    ],
    "basketball": [
        "free throw", "layup", "jump shot", "dribble",
        "crossover", "post move", "defense stance",
        "pass", "rebound",
    ],
    "swimming": [
        "freestyle", "backstroke", "breaststroke", "butterfly",
        "flip turn", "dive start", "kick",
    ],
    "martial_arts": [
        "front kick", "roundhouse kick", "side kick",
        "jab", "cross", "hook", "uppercut",
        "block", "kata", "stance", "combination",
    ],
    "exercise": [
        "squat", "lunge", "plank", "push-up", "jumping jacks",
        "stretching", "leg lifts", "leg kicks", "balance",
        "burpee", "crunch", "mountain climber",
    ],
    "fitness": [
        "squat", "lunge", "plank", "push-up", "jumping jacks",
        "stretching", "leg lifts", "leg kicks", "balance",
        "burpee", "crunch", "mountain climber",
    ],
}

VALID_ANGLES = {"front", "side", "rear", "overhead", "dynamic", "diagonal"}
VALID_SKILL_LEVELS = {"beginner", "intermediate", "professional", "advanced"}


def get_common_techniques(sport):
    """Return the known technique list for a sport."""
    key = sport.lower().replace(" ", "_")
    return SPORT_TECHNIQUES.get(key, [])


def parse_video_description(raw_text):
    """Parse a Pegasus video description into structured fields."""
    result = {
        "technique": "unknown",
        "angle": "unknown",
        "skill_level": "unknown",
        "phases_visible": [],
        "duration_quality": "unknown",
        "summary": "",
    }

    for line in raw_text.strip().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue

        key_part, _, val_part = line.partition(":")
        key = key_part.strip().upper()
        val = val_part.strip()

        if key == "TECHNIQUE":
            result["technique"] = val.lower()
        elif key == "ANGLE":
            result["angle"] = val.lower()
        elif key in ("SKILL LEVEL", "SKILL_LEVEL"):
            result["skill_level"] = val.lower()
        elif key in ("PHASES VISIBLE", "PHASES_VISIBLE"):
            result["phases_visible"] = [
                p.strip() for p in val.split(",") if p.strip()
            ]
        elif key in ("DURATION QUALITY", "DURATION_QUALITY"):
            result["duration_quality"] = val
        elif key == "SUMMARY":
            result["summary"] = val

    result["technique"] = _normalize_technique(result["technique"])
    result["angle"] = _normalize_angle(result["angle"])
    result["skill_level"] = _normalize_skill_level(result["skill_level"])

    return result


def _normalize_technique(val):
    """Extract a short technique name from potentially verbose Pegasus output."""
    known = [
        "jab", "cross", "hook", "uppercut", "combination", "slip",
        "bob and weave", "parry", "block", "footwork", "stance",
        "forehand", "backhand", "serve", "volley", "squat", "deadlift",
        "bench press", "overhead press", "clean", "snatch", "layup",
        "free throw", "freestyle", "backstroke", "breaststroke",
        "front kick", "roundhouse kick", "side kick",
    ]
    val_lower = val.lower()
    for k in known:
        if k in val_lower:
            return k
    words = val_lower.split()
    return " ".join(words[:3]) if words else "unknown"


def _normalize_angle(val):
    """Extract a short angle name from potentially verbose output."""
    valid = {"front", "side", "rear", "overhead", "dynamic", "diagonal"}
    val_lower = val.lower()
    for a in valid:
        if a in val_lower:
            return a
    return val_lower.split(",")[0].strip()[:20] if val else "unknown"


def _normalize_skill_level(val):
    """Extract a short skill level from potentially verbose output."""
    val_lower = val.lower()
    if "professional" in val_lower or "pro " in val_lower or "expert" in val_lower:
        return "professional"
    if "intermediate" in val_lower:
        return "intermediate"
    if "beginner" in val_lower or "novice" in val_lower:
        return "beginner"
    if "advanced" in val_lower:
        return "advanced"
    return val_lower.split(",")[0].strip()[:20] if val else "unknown"


def _find_near_duplicates_via_search(
    descriptions, dataset, client, index_id, threshold_rank=3
):
    """Find near-duplicate videos by searching the index with each
    video's description and checking if other videos appear as top matches."""
    near_dupes = []
    seen_pairs = set()
    samples_list = list(dataset)

    for i, desc in enumerate(descriptions):
        query_text = desc.get("summary", "")
        if not query_text:
            continue

        try:
            results = client.search.query(
                index_id=index_id,
                query_text=query_text,
                search_options=["visual"],
                page_limit=5,
            )

            source_video_id = samples_list[i]["tl_video_id"] if samples_list[i].has_field("tl_video_id") else ""
            for item in results:
                if item.video_id == source_video_id:
                    continue
                for j, other_sample in enumerate(samples_list):
                    other_vid = other_sample["tl_video_id"] if other_sample.has_field("tl_video_id") else ""
                    if j != i and other_vid == item.video_id:
                        pair_key = tuple(sorted([i, j]))
                        if pair_key not in seen_pairs:
                            seen_pairs.add(pair_key)
                            near_dupes.append({
                                "video_a": desc.get("filepath", f"video_{i}"),
                                "video_b": descriptions[j].get("filepath", f"video_{j}"),
                                "note": f"Matched within top {threshold_rank} search results",
                            })
                        break
                if item.rank and item.rank > threshold_rank:
                    break
        except Exception:
            continue

    return near_dupes


def _find_near_duplicates_text(descriptions, threshold=0.6):
    """Fallback near-duplicate detection using Jaccard on word sets."""
    near_dupes = []

    def _word_set(desc):
        text = (
            desc.get("summary", "") + " "
            + desc.get("technique", "") + " "
            + " ".join(desc.get("phases_visible", []))
        )
        return set(text.lower().split())

    word_sets = [_word_set(d) for d in descriptions]

    for i in range(len(descriptions)):
        for j in range(i + 1, len(descriptions)):
            if not word_sets[i] or not word_sets[j]:
                continue
            intersection = word_sets[i] & word_sets[j]
            union = word_sets[i] | word_sets[j]
            jaccard = len(intersection) / len(union) if union else 0
            if jaccard >= threshold:
                near_dupes.append({
                    "video_a": descriptions[i].get("filepath", f"video_{i}"),
                    "video_b": descriptions[j].get("filepath", f"video_{j}"),
                    "similarity": round(jaccard, 3),
                })

    return near_dupes


def _generate_recommendations(
    techniques, angles, skill_levels, near_dupes, coverage_gaps, sport
):
    """Generate actionable recommendations from the profile analysis."""
    recs = []

    for gap in coverage_gaps:
        if "Missing technique" in gap:
            tech = gap.replace("Missing technique: ", "")
            recs.append(f"Add reference videos for '{tech}' technique.")

    if len(angles) == 1:
        sole_angle = list(angles.keys())[0]
        recs.append(
            f"All videos shot from {sole_angle} angle. "
            f"Add side and/or overhead angles for rotation and depth analysis."
        )
    elif len(angles) == 2 and "front" in angles and "side" not in angles:
        recs.append("Add side-angle footage for better lateral movement analysis.")

    level_keys = {k.lower() for k in skill_levels.keys()}
    if "professional" not in level_keys and "pro" not in level_keys:
        recs.append(
            "No professional-level references found. "
            "Add pro footage for a stronger coaching baseline."
        )
    if "beginner" not in level_keys:
        recs.append(
            "No beginner-level references. Adding some can help "
            "illustrate common mistakes for contrast."
        )

    if near_dupes:
        recs.append(
            f"{len(near_dupes)} near-duplicate pair(s) detected. "
            f"Consider removing redundant clips to save indexing quota."
        )

    total = sum(techniques.values())
    if total < 3:
        recs.append(
            f"Only {total} reference video(s). Aim for 5+ clips "
            f"covering different techniques and angles."
        )

    if techniques:
        max_count = max(techniques.values())
        min_count = min(techniques.values())
        if max_count > 3 * min_count and len(techniques) > 1:
            over = max(techniques, key=techniques.get)
            under = min(techniques, key=techniques.get)
            recs.append(
                f"Imbalanced coverage: '{over}' has {techniques[over]} clips "
                f"vs '{under}' with {techniques[under]}. Balance the library."
            )

    return recs


def profile_references(dataset, sport, client, similarity_threshold=0.6, ctx=None):
    """Generate a DNA health card for a reference video dataset.

    Args:
        dataset: FiftyOne dataset of reference videos with tl_video_id fields.
        sport: Sport name (e.g. "boxing", "tennis").
        client: Initialized TwelveLabs client.
        similarity_threshold: Threshold for near-duplicate detection (0-1).
        ctx: Optional operator context for progress updates.

    Returns:
        dict - the full TrainingDNA profile.
    """
    descriptions = []
    samples_list = list(dataset)

    # Try to get index_id from the first sample
    index_id = None
    for s in samples_list:
        if s.has_field("tl_index_id"):
            index_id = s["tl_index_id"]
            break

    # Step 1: Describe each video with Pegasus
    for idx, sample in enumerate(samples_list):
        if ctx:
            ctx.set_progress(
                label=f"Analyzing video {idx + 1}/{len(samples_list)}...",
                progress=0.1 + (0.5 * idx / max(len(samples_list), 1)),
            )

        video_id = sample["tl_video_id"] if sample.has_field("tl_video_id") else None
        if not video_id:
            descriptions.append({
                "technique": "unknown", "angle": "unknown",
                "skill_level": "unknown", "phases_visible": [],
                "duration_quality": "unknown", "summary": "",
                "filepath": sample.filepath if hasattr(sample, "filepath") else f"video_{idx}",
            })
            continue

        desc_prompt = (
            f"Analyze this {sport} technique reference video.\n"
            "Reply with ONLY these fields, one per line. "
            "Use ONLY short keywords, not full sentences.\n\n"
            "TECHNIQUE: <one or two words ONLY, e.g. jab, cross, hook, uppercut, combination>\n"
            "ANGLE: <one word ONLY: front, side, overhead, rear, or dynamic>\n"
            "SKILL LEVEL: <one word ONLY: beginner, intermediate, or professional>\n"
            "PHASES VISIBLE: <short comma-separated list, e.g. stance, extension, recovery>\n"
            "DURATION QUALITY: <yes or no>\n"
            "SUMMARY: <one sentence max>"
        )

        try:
            desc_response = _analyze_with_retry(client, desc_prompt, video_id)
            parsed = parse_video_description(desc_response.data)
        except Exception:
            parsed = {
                "technique": "unknown", "angle": "unknown",
                "skill_level": "unknown", "phases_visible": [],
                "duration_quality": "unknown", "summary": "",
            }

        parsed["filepath"] = (
            sample.filepath if hasattr(sample, "filepath") else f"video_{idx}"
        )
        parsed["sample_id"] = sample.id
        descriptions.append(parsed)

    # Step 2: Compute distributions
    if ctx:
        ctx.set_progress(label="Computing distributions...", progress=0.7)

    techniques = Counter(
        d["technique"] for d in descriptions if d["technique"] != "unknown"
    )
    angles = Counter(
        d["angle"] for d in descriptions if d["angle"] != "unknown"
    )
    skill_levels = Counter(
        d["skill_level"] for d in descriptions if d["skill_level"] != "unknown"
    )

    # Step 3: Find near-duplicates
    if ctx:
        ctx.set_progress(label="Checking for near-duplicates...", progress=0.8)

    if index_id:
        near_dupes = _find_near_duplicates_via_search(
            descriptions, dataset, client, index_id
        )
    else:
        near_dupes = _find_near_duplicates_text(
            descriptions, threshold=similarity_threshold
        )

    # Step 4: Identify coverage gaps
    coverage_gaps = []
    common_techniques = get_common_techniques(sport)
    detected_techniques_lower = {t.lower() for t in techniques.keys()}

    for tech in common_techniques:
        if tech.lower() not in detected_techniques_lower:
            coverage_gaps.append(f"Missing technique: {tech}")

    if len(angles) == 1:
        sole = list(angles.keys())[0]
        coverage_gaps.append(
            f"Only one camera angle ({sole}). "
            f"Add alternative angles for better analysis."
        )

    level_keys_lower = {k.lower() for k in skill_levels.keys()}
    if not (level_keys_lower & {"professional", "pro", "advanced"}):
        coverage_gaps.append(
            "No professional-level references. "
            "Add pro footage for a better coaching baseline."
        )

    # Step 5: Generate recommendations
    recommendations = _generate_recommendations(
        techniques, angles, skill_levels, near_dupes, coverage_gaps, sport
    )

    if ctx:
        ctx.set_progress(label="TrainingDNA profiling complete!", progress=1.0)

    # Step 6: Assemble the health card
    return {
        "sport": sport,
        "total_videos": len(dataset),
        "technique_distribution": dict(techniques),
        "angle_distribution": dict(angles),
        "skill_level_distribution": dict(skill_levels),
        "video_descriptions": [
            {
                "filepath": d.get("filepath", ""),
                "technique": d["technique"],
                "angle": d["angle"],
                "skill_level": d["skill_level"],
                "phases_visible": d["phases_visible"],
                "summary": d["summary"],
            }
            for d in descriptions
        ],
        "near_duplicates": near_dupes,
        "coverage_gaps": coverage_gaps,
        "recommendations": recommendations,
    }
