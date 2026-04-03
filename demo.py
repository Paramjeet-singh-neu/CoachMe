#!/usr/bin/env python3
"""
CoachMe+ Demo Script — run the full pipeline from the terminal.

Usage:
    # Core only
    python demo.py --sport boxing --refs ./demo-videos/reference --athlete ./demo-videos/athlete/my_jab.mp4

    # Full pipeline
    python demo.py --sport boxing --refs ./demo-videos/reference \
                   --athlete ./demo-videos/athlete/my_jab.mp4 \
                   --focus "jab technique" \
                   --countervision --validate --sync --profile \
                   --launch-app
"""

import argparse
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="CoachMe+ — AI Sports Coach Demo")
    parser.add_argument("--sport", required=True, help="Sport or skill name")
    parser.add_argument("--refs", required=True, help="Path to reference videos folder")
    parser.add_argument("--athlete", required=True, help="Path to athlete video")
    parser.add_argument("--focus", default="overall technique", help="Focus area for coaching")
    parser.add_argument("--countervision", action="store_true", help="Run CounterVision — find correct form")
    parser.add_argument("--validate", action="store_true", help="Run CoachCheck — hallucination detection")
    parser.add_argument("--sync", action="store_true", help="Run TechniqueSync — phase alignment")
    parser.add_argument("--sync-ref", default=None, help="Specific reference video filename for TechniqueSync")
    parser.add_argument("--profile", action="store_true", help="Run TrainingDNA — reference library profiling")
    parser.add_argument("--launch-app", action="store_true", help="Launch FiftyOne App after analysis")
    args = parser.parse_args()

    if not os.path.isdir(args.refs):
        print(f"Error: reference folder not found: {args.refs}")
        sys.exit(1)
    if not os.path.isfile(args.athlete):
        print(f"Error: athlete video not found: {args.athlete}")
        sys.exit(1)

    import numpy as np
    import fiftyone as fo
    from fiftyone import ViewField as F
    from twelvelabs import TwelveLabs
    from twelvelabs.indexes.types import IndexesCreateRequestModelsItem

    VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm")

    def get_video_embedding(client, video_path):
        """Get a 512-dim Marengo embedding vector for a video."""
        with open(video_path, "rb") as f:
            task = client.embed.tasks.create(
                model_name="marengo3.0",
                video_file=f,
                video_embedding_scope=["clip", "video"],
            )
        client.embed.tasks.wait_for_done(task.id)
        result = client.embed.tasks.retrieve(task.id, embedding_option="visual")
        for seg in result.video_embedding.segments:
            if seg.embedding_scope == "video":
                return np.array(seg.float_)
        return np.array(result.video_embedding.segments[0].float_)

    def cosine_similarity(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # ── Connect ──
    client = TwelveLabs(api_key=os.getenv("TWELVELABS_API_KEY"))
    print("Connected to Twelve Labs")

    # ── Index ──
    index_name = "coachme-index"
    index_id = None
    for idx in client.indexes.list():
        if idx.index_name == index_name:
            index_id = idx.id
            break
    if not index_id:
        idx = client.indexes.create(
            index_name=index_name,
            models=[
                IndexesCreateRequestModelsItem(model_name="marengo3.0", model_options=["visual", "audio"]),
                IndexesCreateRequestModelsItem(model_name="pegasus1.2", model_options=["visual", "audio"]),
            ],
        )
        index_id = idx.id
    print(f"Index: {index_id}")

    def upload_wait(path, label="Video"):
        with open(path, "rb") as f:
            task = client.tasks.create(index_id=index_id, video_file=f)
        def _cb(t):
            print(f"  [{label}] {t.status}", end="\r")
        result = client.tasks.wait_for_done(task.id, callback=_cb)
        print(f"  [{label}] ready — video_id={result.video_id}      ")
        return result.video_id

    # ══════════════════════════════════════════
    # STEP 1: Reference videos
    # ══════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  STEP 1: Loading reference videos for {args.sport}")
    print(f"{'='*60}")

    ds_name = f"coachme-reference-{args.sport}"
    if fo.dataset_exists(ds_name):
        ref_dataset = fo.load_dataset(ds_name)
    else:
        ref_dataset = fo.Dataset(ds_name, persistent=True)
        ref_dataset.tags = ["coachme", "reference", args.sport]

    ref_videos = sorted(
        os.path.join(args.refs, f)
        for f in os.listdir(args.refs)
        if f.lower().endswith(VIDEO_EXT) and not f.startswith(".")
    )
    print(f"Found {len(ref_videos)} reference videos")

    ref_lookup = {}  # video_id -> {filepath, embedding}
    for s in ref_dataset:
        vid = s.get_field("tl_video_id") if hasattr(s, "get_field") else s.get("tl_video_id")
        emb = s.get_field("tl_embedding") if hasattr(s, "get_field") else s.get("tl_embedding")
        if vid:
            ref_lookup[vid] = {"filepath": s.filepath, "embedding": emb}

    for vpath in ref_videos:
        existing = ref_dataset.match(F("filepath") == vpath)
        if len(existing) > 0:
            print(f"  Skipping (already indexed): {os.path.basename(vpath)}")
            continue
        try:
            vid = upload_wait(vpath, os.path.basename(vpath))
            print(f"  Embedding {os.path.basename(vpath)}...")
            embedding = get_video_embedding(client, vpath)
            sample = fo.Sample(filepath=vpath)
            sample["sport"] = args.sport
            sample["role"] = "reference"
            sample["tl_video_id"] = vid
            sample["tl_index_id"] = index_id
            sample["tl_embedding"] = embedding.tolist()
            ref_dataset.add_sample(sample)
            ref_lookup[vid] = {"filepath": vpath, "embedding": embedding}
        except Exception as e:
            print(f"  ERROR uploading {os.path.basename(vpath)}: {e}")
            continue

    ref_dataset.save()
    print(f"Reference dataset: {len(ref_dataset)} samples")

    # ══════════════════════════════════════════
    # STEP 1.5: TrainingDNA (if requested)
    # ══════════════════════════════════════════
    if args.profile:
        print(f"\n{'='*60}")
        print(f"  STEP 1.5: TrainingDNA — Profiling reference library")
        print(f"{'='*60}")

        from coachme.training_dna import profile_references as run_profile
        profile = run_profile(ref_dataset, args.sport, client)

        ref_dataset.info["training_dna"] = profile
        ref_dataset.save()

        print(f"  Total videos: {profile['total_videos']}")
        print(f"  Techniques: {profile['technique_distribution']}")
        print(f"  Angles: {profile['angle_distribution']}")
        if profile["near_duplicates"]:
            print(f"  Near-duplicates: {len(profile['near_duplicates'])}")
            for nd in profile["near_duplicates"]:
                print(f"    {os.path.basename(nd['video_a'])} <-> {os.path.basename(nd['video_b'])}: {nd['similarity']}%")
        if profile["coverage_gaps"]:
            print(f"  Gaps:")
            for gap in profile["coverage_gaps"]:
                print(f"    - {gap}")
        print(f"  Recommendations:")
        for rec in profile["recommendations"]:
            print(f"    - {rec}")

    # ══════════════════════════════════════════
    # STEP 2: Upload athlete video
    # ══════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  STEP 2: Uploading athlete video")
    print(f"{'='*60}")

    athlete_video_id = upload_wait(args.athlete, "Athlete")

    # ══════════════════════════════════════════
    # STEP 3: Marengo embedding similarity
    # ══════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  STEP 3: Computing technique similarity (Marengo Embeddings)")
    print(f"{'='*60}")

    print("  Embedding athlete video...")
    athlete_embedding = get_video_embedding(client, args.athlete)

    similarity_scores = []
    top_matches = []
    for vid_id, ref_info in ref_lookup.items():
        ref_emb = ref_info.get("embedding")
        if ref_emb is None:
            continue
        ref_emb = np.array(ref_emb) if not isinstance(ref_emb, np.ndarray) else ref_emb
        sim = cosine_similarity(athlete_embedding, ref_emb)
        pct = round(sim * 100, 1)
        similarity_scores.append(pct)
        top_matches.append({
            "reference_video_id": vid_id,
            "reference_filepath": ref_info["filepath"],
            "similarity_pct": pct,
        })
        print(f"  {os.path.basename(ref_info['filepath'])}: {pct}%")

    top_matches.sort(key=lambda m: m["similarity_pct"], reverse=True)

    avg_score = round(sum(similarity_scores) / len(similarity_scores), 1) if similarity_scores else 0.0
    print(f"\n  Average similarity: {avg_score}%")

    # ══════════════════════════════════════════
    # STEP 4: Pegasus coaching
    # ══════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  STEP 4: Generating AI coaching feedback (Pegasus)")
    print(f"{'='*60}")

    prompt = (
        f"You are an elite {args.sport} coach reviewing an athlete's technique video.\n"
        f"Focus area: {args.focus}.\n\n"
        f"Provide your analysis in this exact format:\n\n"
        f"## Technique Score: [give a score out of 10]\n\n"
        f"## What You're Doing Well\n"
        f"- [timestamp] observation\n\n"
        f"## What Needs Work\n"
        f"- [timestamp] specific issue and how to fix it\n\n"
        f"## Drill Prescription\n"
        f"List 3 specific drills the athlete should do to improve.\n\n"
        f"Be direct, specific, and encouraging. Use timestamps like [0:04] throughout."
    )

    pegasus_result = client.analyze(video_id=athlete_video_id, prompt=prompt)
    coaching_text = getattr(pegasus_result, "data", None) or str(pegasus_result)
    if not isinstance(coaching_text, str):
        coaching_text = str(coaching_text)

    print(coaching_text)

    # ══════════════════════════════════════════
    # STEP 5: Save core results to FiftyOne
    # ══════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  STEP 5: Saving to FiftyOne")
    print(f"{'='*60}")

    if fo.dataset_exists("coachme-athletes"):
        athlete_ds = fo.load_dataset("coachme-athletes")
    else:
        athlete_ds = fo.Dataset("coachme-athletes", persistent=True)
        athlete_ds.tags = ["coachme", "athletes"]

    sample = fo.Sample(filepath=args.athlete)
    sample["sport"] = args.sport
    sample["role"] = "athlete"
    sample["focus_area"] = args.focus
    sample["tl_video_id"] = athlete_video_id
    sample["tl_index_id"] = index_id
    sample["similarity_score"] = avg_score
    sample["reference_matches"] = top_matches
    sample["coaching_feedback"] = coaching_text
    sample["analyzed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    athlete_ds.add_sample(sample)
    athlete_ds.save()

    athlete_sample_id = sample.id
    print(f"  Saved sample: {athlete_sample_id}")

    # ══════════════════════════════════════════
    # STEP 6: CoachCheck (if requested)
    # ══════════════════════════════════════════
    if args.validate:
        print(f"\n{'='*60}")
        print(f"  STEP 6: CoachCheck — Validating coaching claims")
        print(f"{'='*60}")

        from coachme.coach_check import validate_coaching as run_validate
        validation = run_validate(sample, client)

        sample["coaching_validation"] = validation
        sample.save()

        g_score = validation["grounding_score"]
        h_count = validation["hallucinations_flagged"]
        total_claims = len(validation["claims"])
        print(f"  Grounding score: {g_score}%")
        print(f"  Claims verified: {total_claims - h_count}/{total_claims}")
        print(f"  Hallucinations flagged: {h_count}")
        for claim in validation["claims"]:
            status = "GROUNDED" if claim["is_grounded"] else "HALLUCINATION"
            print(f"    [{status}] [{claim['timestamp']}] {claim['original_claim']}")

    # ══════════════════════════════════════════
    # STEP 7: CounterVision (if requested)
    # ══════════════════════════════════════════
    if args.countervision:
        print(f"\n{'='*60}")
        print(f"  STEP 7: CounterVision — Finding correct form")
        print(f"{'='*60}")

        from coachme.countervision import find_correct_form as run_cv
        cv_matches = run_cv(sample, index_id, client)

        sample["counterfactual_matches"] = cv_matches
        sample.save()

        print(f"  Found {len(cv_matches)} problem→correction matches:")
        for m in cv_matches:
            print(f"    [{m.get('timestamp', '?')}] {m.get('problem_description', '?')}")
            print(f"      Correct: {m.get('correct_description', '?')}")
            print(f"      Confidence: {m.get('confidence', 0)}%")

    # ══════════════════════════════════════════
    # STEP 8: TechniqueSync (if requested)
    # ══════════════════════════════════════════
    if args.sync:
        print(f"\n{'='*60}")
        print(f"  STEP 8: TechniqueSync — Phase alignment")
        print(f"{'='*60}")

        from coachme.technique_sync import sync_technique as run_sync

        # Pick reference sample to sync against
        ref_sample = None
        for s in ref_dataset:
            vid = s.get_field("tl_video_id") if hasattr(s, "get_field") else s.get("tl_video_id")
            if vid:
                if args.sync_ref and args.sync_ref not in s.filepath:
                    continue
                ref_sample = s
                break

        if ref_sample:
            alignment = run_sync(sample, ref_sample, args.sport, client)
            sample["phase_alignment"] = alignment
            sample.save()

            print(f"  Sync score: {alignment['overall_sync_score']:.3f}")
            print(f"  Phases aligned: {len(alignment['phases'])}")
            for p in alignment["phases"]:
                delta = p["timing_delta_seconds"]
                direction = "slower" if delta > 0 else "faster" if delta < 0 else "same"
                print(f"    {p['athlete_phase']}: {p['athlete_time']} vs ref {p['reference_time']} ({abs(delta):.2f}s {direction})")
        else:
            print("  No reference sample found for sync")

    # ══════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"         COACHME+ — ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"  Sport:           {args.sport}")
    print(f"  Focus:           {args.focus}")
    print(f"  References:      {len(ref_lookup)} videos")
    print(f"  Similarity:      {avg_score}%")
    print(f"  Feedback:        {len(coaching_text)} chars")

    if args.validate and "coaching_validation" in sample.field_names:
        v = sample["coaching_validation"]
        print(f"  CoachCheck:      {v['grounding_score']}% grounded, {v['hallucinations_flagged']} hallucinations")
    if args.countervision and "counterfactual_matches" in sample.field_names:
        print(f"  CounterVision:   {len(sample['counterfactual_matches'])} matches")
    if args.sync and "phase_alignment" in sample.field_names:
        print(f"  TechniqueSync:   {sample['phase_alignment']['overall_sync_score']:.3f} sync score")
    if args.profile:
        print(f"  TrainingDNA:     profiled")

    print(f"{'='*60}")
    print(f'\n"Elite athletes have coaches. Everyone else has CoachMe+."')

    if args.launch_app:
        print("\nLaunching FiftyOne App...")
        session = fo.launch_app(athlete_ds)
        print("Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
