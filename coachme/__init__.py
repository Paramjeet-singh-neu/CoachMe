"""
CoachMe+ — AI Sports Coach FiftyOne Plugin
Built by Aatmaj Rajore & Paramjeet Singh
Voxel51 + Twelve Labs Hackathon, April 2026

Core: Marengo 2.7 (similarity) + Pegasus 1.2 (coaching feedback)
CounterVision: Auto-find correct form for each problem
TechniqueSync: Temporal phase alignment athlete vs pro
CoachCheck: Hallucination detection on AI coaching output
TrainingDNA: Reference library health card
"""

import os
import time
import numpy as np
import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone as fo
from fiftyone import ViewField as F
from twelvelabs import TwelveLabs
from twelvelabs.indexes.types import IndexesCreateRequestModelsItem
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def get_client():
    api_key = os.getenv("TWELVELABS_API_KEY")
    if not api_key:
        raise ValueError("Missing TWELVELABS_API_KEY — add it to your .env file")
    return TwelveLabs(api_key=api_key)


def get_or_create_index(client, index_name="coachme-index"):
    """Get existing index or create a new one with Marengo + Pegasus."""
    for idx in client.indexes.list():
        if idx.index_name == index_name:
            return idx.id

    idx = client.indexes.create(
        index_name=index_name,
        models=[
            IndexesCreateRequestModelsItem(model_name="marengo3.0", model_options=["visual", "audio"]),
            IndexesCreateRequestModelsItem(model_name="pegasus1.2", model_options=["visual", "audio"]),
        ],
    )
    return idx.id


def upload_and_wait(client, index_id, video_path, ctx=None, label="Video", timeout_s=600):
    """Upload a video and wait until indexing completes."""
    with open(video_path, "rb") as f:
        task = client.tasks.create(index_id=index_id, video_file=f)
    if ctx:
        ctx.set_progress(label=f"{label}: indexing...", progress=None)

    def _on_status(t):
        if ctx:
            ctx.set_progress(label=f"{label}: {t.status}", progress=None)

    result = client.tasks.wait_for_done(task.id, callback=_on_status)
    if result.status != "ready":
        raise RuntimeError(f"Twelve Labs indexing failed for {video_path}: {result.status}")
    return result.video_id


VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def find_videos(directory):
    """Return sorted list of video file paths in a directory."""
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(VIDEO_EXTENSIONS) and not f.startswith(".")
    )


def _get_sample_field(sample, field, default=None):
    """Safely get a field from a FiftyOne sample."""
    try:
        val = sample.get_field(field)
        return val if val is not None else default
    except Exception:
        return default


def get_video_embedding(client, video_path):
    """Get a 512-dim Marengo embedding vector for a video file."""
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
            return seg.float_
    return result.video_embedding.segments[0].float_


def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two vectors."""
    a, b = np.array(vec_a), np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ─────────────────────────────────────────────
# Operator 1: Load Reference Videos (Core)
# ─────────────────────────────────────────────

class LoadReferenceVideos(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="load_reference_videos",
            label="CoachMe+: Load Reference Videos",
            description="Index reference technique videos as the coaching baseline",
            icon="/assets/icon.svg",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str(
            "sport",
            label="Sport or Skill",
            description="e.g. boxing, squat, tennis serve, deadlift",
            required=True,
        )
        inputs.str(
            "videos_dir",
            label="Path to Reference Videos Folder",
            description="Folder containing your reference technique videos (.mp4, .mov, .avi)",
            required=True,
        )

        videos_dir = ctx.params.get("videos_dir", "")
        if videos_dir and os.path.isdir(videos_dir):
            vids = find_videos(videos_dir)
            inputs.view(
                "preview",
                types.Notice(
                    label=f"Found {len(vids)} video(s) in this folder"
                    if vids
                    else "No video files found in this folder"
                ),
            )

        return types.Property(inputs)

    def execute(self, ctx):
        sport = ctx.params.get("sport", "general").strip().lower()
        videos_dir = ctx.params.get("videos_dir", "").strip()

        if not videos_dir or not os.path.isdir(videos_dir):
            return {"error": f"Directory not found: {videos_dir}"}

        video_files = find_videos(videos_dir)
        if not video_files:
            return {"error": f"No video files found in {videos_dir}"}

        client = get_client()
        index_id = get_or_create_index(client)

        ds_name = f"coachme-reference-{sport}"
        if fo.dataset_exists(ds_name):
            dataset = fo.load_dataset(ds_name)
        else:
            dataset = fo.Dataset(ds_name, persistent=True)
            dataset.tags = ["coachme", "reference", sport]

        indexed = 0
        skipped = 0
        errors = []

        for i, fpath in enumerate(video_files):
            fname = os.path.basename(fpath)
            ctx.set_progress(
                label=f"Indexing {fname} ({i + 1}/{len(video_files)})",
                progress=(i / len(video_files)),
            )

            existing = dataset.match(F("filepath") == fpath)
            if len(existing) > 0:
                skipped += 1
                continue

            try:
                video_id = upload_and_wait(
                    client, index_id, fpath, ctx=ctx, label=f"Uploading {fname}"
                )
                ctx.set_progress(label=f"Embedding {fname}...", progress=(i / len(video_files)))
                embedding = get_video_embedding(client, fpath)
                sample = fo.Sample(filepath=fpath)
                sample["sport"] = sport
                sample["role"] = "reference"
                sample["tl_video_id"] = video_id
                sample["tl_index_id"] = index_id
                sample["tl_embedding"] = embedding
                dataset.add_sample(sample)
                indexed += 1
            except Exception as e:
                errors.append(f"{fname}: {e}")

        dataset.save()
        ctx.set_progress(label="Done!", progress=1.0)
        ctx.trigger("open_dataset", {"name": ds_name})

        return {
            "status": "done",
            "indexed": indexed,
            "skipped": skipped,
            "errors": len(errors),
            "error_details": "; ".join(errors) if errors else "",
            "dataset": ds_name,
            "index_id": index_id,
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("status", label="Status")
        outputs.int("indexed", label="Videos Indexed")
        outputs.int("skipped", label="Already Indexed (Skipped)")
        outputs.int("errors", label="Errors")
        outputs.str("error_details", label="Error Details")
        outputs.str("dataset", label="Dataset Name")
        outputs.str("index_id", label="Twelve Labs Index ID")
        return types.Property(outputs)


# ─────────────────────────────────────────────
# Operator 2: Analyze Technique (Core)
# ─────────────────────────────────────────────

class AnalyzeTechnique(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="analyze_technique",
            label="CoachMe+: Analyze My Technique",
            description="Compare your video to references and get AI coaching feedback",
            icon="/assets/icon.svg",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str(
            "sport",
            label="Sport or Skill",
            description="Must match what you used when loading references",
            required=True,
        )
        inputs.str(
            "video_path",
            label="Path to Your Video",
            description="Full path to the video you want analyzed",
            required=True,
        )
        inputs.str(
            "focus",
            label="What to Focus On",
            description="e.g. elbow position, footwork, hip rotation (optional)",
            required=False,
        )

        video_path = ctx.params.get("video_path", "")
        if video_path and not os.path.isfile(video_path):
            inputs.view("warn", types.Warning(label=f"File not found: {video_path}"))

        return types.Property(inputs)

    def execute(self, ctx):
        sport = ctx.params.get("sport", "").strip().lower()
        video_path = ctx.params.get("video_path", "").strip()
        focus = ctx.params.get("focus", "").strip() or "overall technique"

        if not os.path.isfile(video_path):
            return {
                "similarity_score": 0.0,
                "coaching_feedback": f"Error: video not found at {video_path}",
                "video_id": "",
            }

        ds_name = f"coachme-reference-{sport}"
        if not fo.dataset_exists(ds_name):
            return {
                "similarity_score": 0.0,
                "coaching_feedback": f"Error: no reference videos for '{sport}'. Run 'Load Reference Videos' first.",
                "video_id": "",
            }

        ref_dataset = fo.load_dataset(ds_name)
        ref_samples = []
        for s in ref_dataset:
            vid = _get_sample_field(s, "tl_video_id")
            emb = _get_sample_field(s, "tl_embedding")
            if vid and emb:
                ref_samples.append({"video_id": vid, "filepath": s.filepath, "embedding": emb})

        if not ref_samples:
            return {
                "similarity_score": 0.0,
                "coaching_feedback": "Error: reference dataset has no indexed videos.",
                "video_id": "",
            }

        client = get_client()
        index_id = get_or_create_index(client)

        # Step 1: Upload athlete video
        ctx.set_progress(label="Uploading your video to Twelve Labs...", progress=0.1)
        athlete_video_id = upload_and_wait(
            client, index_id, video_path, ctx=ctx, label="Uploading athlete video"
        )

        # Step 2: Compute Marengo embeddings and cosine similarity
        ctx.set_progress(label="Computing video embedding (Marengo)...", progress=0.35)
        athlete_embedding = get_video_embedding(client, video_path)

        ctx.set_progress(label="Comparing against references...", progress=0.5)
        similarity_scores = []
        top_matches = []
        for ref in ref_samples:
            sim = cosine_similarity(athlete_embedding, ref["embedding"])
            pct = round(sim * 100, 1)
            similarity_scores.append(pct)
            top_matches.append({
                "reference_video_id": ref["video_id"],
                "reference_filepath": ref["filepath"],
                "similarity_pct": pct,
            })

        top_matches.sort(key=lambda m: m["similarity_pct"], reverse=True)
        avg_score = round(sum(similarity_scores) / len(similarity_scores), 1) if similarity_scores else 0.0

        # Step 3: Pegasus coaching feedback
        ctx.set_progress(label="Generating AI coaching feedback with Pegasus...", progress=0.7)

        prompt = (
            f"You are an elite {sport} coach reviewing an athlete's technique video.\n"
            f"Focus area: {focus}.\n\n"
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

        pegasus_result = client.analyze(
            video_id=athlete_video_id,
            prompt=prompt,
        )
        coaching_text = getattr(pegasus_result, "data", None) or str(pegasus_result)
        if not isinstance(coaching_text, str):
            coaching_text = str(coaching_text)

        # Step 4: Write results to FiftyOne
        ctx.set_progress(label="Saving results to FiftyOne...", progress=0.9)

        if fo.dataset_exists("coachme-athletes"):
            athlete_ds = fo.load_dataset("coachme-athletes")
        else:
            athlete_ds = fo.Dataset("coachme-athletes", persistent=True)
            athlete_ds.tags = ["coachme", "athletes"]

        sample = fo.Sample(filepath=video_path)
        sample["sport"] = sport
        sample["role"] = "athlete"
        sample["focus_area"] = focus
        sample["tl_video_id"] = athlete_video_id
        sample["tl_index_id"] = index_id
        sample["similarity_score"] = avg_score
        sample["reference_matches"] = top_matches
        sample["coaching_feedback"] = coaching_text
        sample["analyzed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        athlete_ds.add_sample(sample)
        athlete_ds.save()

        ctx.set_progress(label="Analysis complete!", progress=1.0)
        ctx.trigger("open_dataset", {"name": "coachme-athletes"})

        return {
            "similarity_score": avg_score,
            "coaching_feedback": coaching_text,
            "video_id": athlete_video_id,
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.float("similarity_score", label="Technique Similarity (%)")
        outputs.str("coaching_feedback", label="AI Coaching Feedback")
        outputs.str("video_id", label="Twelve Labs Video ID")
        return types.Property(outputs)


# ─────────────────────────────────────────────
# Operator 3: View Coaching Report (Core)
# ─────────────────────────────────────────────

class ViewCoachingReport(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="view_coaching_report",
            label="CoachMe+: View Coaching Report",
            description="Browse coaching results for analyzed athletes",
            icon="/assets/icon.svg",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        if not fo.dataset_exists("coachme-athletes"):
            inputs.view(
                "no_data",
                types.Warning(label="No athlete analyses found. Run 'Analyze My Technique' first."),
            )
            return types.Property(inputs)

        ds = fo.load_dataset("coachme-athletes")
        if len(ds) == 0:
            inputs.view(
                "empty",
                types.Warning(label="Dataset is empty. Analyze a video first."),
            )
            return types.Property(inputs)

        choices = types.DropdownView()
        for sample in ds:
            sport = _get_sample_field(sample, "sport", "?")
            score = _get_sample_field(sample, "similarity_score", 0)
            ts = _get_sample_field(sample, "analyzed_at", "")
            label = f"{sport.upper()} | Score: {score}% | {ts} | {os.path.basename(sample.filepath)}"
            choices.add_choice(sample.id, label=label)

        inputs.str(
            "sample_id",
            label="Select an Analysis",
            description="Pick a previously analyzed video to view its coaching report",
            required=True,
            view=choices,
        )

        sample_id = ctx.params.get("sample_id")
        if sample_id:
            try:
                sample = ds[sample_id]
                sport = _get_sample_field(sample, "sport", "")
                score = _get_sample_field(sample, "similarity_score", 0)
                focus = _get_sample_field(sample, "focus_area", "")
                feedback = _get_sample_field(sample, "coaching_feedback", "")
                matches = _get_sample_field(sample, "reference_matches", [])
                analyzed = _get_sample_field(sample, "analyzed_at", "")
                validation = _get_sample_field(sample, "coaching_validation")
                counterfactuals = _get_sample_field(sample, "counterfactual_matches")
                phase_align = _get_sample_field(sample, "phase_alignment")

                header = (
                    f"**Sport:** {sport.upper()}  |  "
                    f"**Similarity:** {score}%  |  "
                    f"**Focus:** {focus}  |  "
                    f"**Analyzed:** {analyzed}"
                )
                inputs.view("header", types.Notice(label=header))

                if matches:
                    match_lines = "\n".join(
                        f"  - {os.path.basename(m.get('reference_filepath', '?'))}: {m.get('similarity_pct', 0)}%"
                        for m in matches
                    )
                    inputs.view("matches_info", types.Notice(label=f"**Reference Matches:**\n{match_lines}"))

                inputs.view("feedback_display", types.Notice(label=f"**Coaching Feedback:**\n\n{feedback}"))

                # Show CoachCheck results if available
                if validation:
                    g_score = validation.get("grounding_score", 0)
                    h_count = validation.get("hallucinations_flagged", 0)
                    total_claims = len(validation.get("claims", []))
                    inputs.view(
                        "validation_info",
                        types.Notice(
                            label=f"**CoachCheck Validation:** {g_score}% grounded | "
                            f"{total_claims - h_count}/{total_claims} claims verified | "
                            f"{h_count} hallucination(s) flagged"
                        ),
                    )

                # Show CounterVision results if available
                if counterfactuals:
                    cf_lines = []
                    for cf in counterfactuals:
                        cf_lines.append(
                            f"  - [{cf.get('timestamp', '?')}] Problem: {cf.get('problem_description', '?')}\n"
                            f"    Correct: {cf.get('correct_description', '?')} "
                            f"(confidence: {cf.get('confidence', 0)}%)"
                        )
                    inputs.view(
                        "counterfactual_info",
                        types.Notice(label=f"**CounterVision Matches:**\n" + "\n".join(cf_lines)),
                    )

                # Show TechniqueSync results if available
                if phase_align:
                    sync_score = phase_align.get("overall_sync_score", 0)
                    phases = phase_align.get("phases", [])
                    phase_lines = []
                    for p in phases:
                        delta = p.get("timing_delta_seconds", 0)
                        direction = "slower" if delta > 0 else "faster" if delta < 0 else "same"
                        phase_lines.append(
                            f"  - {p.get('athlete_phase', '?')}: {p.get('athlete_time', '?')} "
                            f"vs ref {p.get('reference_time', '?')} ({abs(delta):.2f}s {direction})"
                        )
                    inputs.view(
                        "sync_info",
                        types.Notice(
                            label=f"**TechniqueSync:** Overall sync {sync_score:.1%}\n" + "\n".join(phase_lines)
                        ),
                    )

            except Exception:
                inputs.view("err", types.Warning(label="Could not load this sample."))

        return types.Property(inputs)

    def execute(self, ctx):
        sample_id = ctx.params.get("sample_id")
        if sample_id and fo.dataset_exists("coachme-athletes"):
            ctx.trigger("open_dataset", {"name": "coachme-athletes"})
        return {"status": "report_viewed", "sample_id": sample_id or ""}

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("status", label="Status")
        outputs.str("sample_id", label="Sample ID")
        return types.Property(outputs)


# ─────────────────────────────────────────────
# Operator 4: Find Correct Form (CounterVision)
# ─────────────────────────────────────────────

class FindCorrectForm(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="find_correct_form",
            label="CoachMe+: CounterVision — Find Correct Form",
            description="Auto-find reference video showing correct form for each identified problem",
            icon="/assets/icon.svg",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        if not fo.dataset_exists("coachme-athletes"):
            inputs.view("no_data", types.Warning(label="No athlete analyses found. Run 'Analyze My Technique' first."))
            return types.Property(inputs)

        ds = fo.load_dataset("coachme-athletes")
        if len(ds) == 0:
            inputs.view("empty", types.Warning(label="No analyzed videos. Run 'Analyze My Technique' first."))
            return types.Property(inputs)

        choices = types.DropdownView()
        for sample in ds:
            sport = _get_sample_field(sample, "sport", "?")
            score = _get_sample_field(sample, "similarity_score", 0)
            label = f"{sport.upper()} | Score: {score}% | {os.path.basename(sample.filepath)}"
            choices.add_choice(sample.id, label=label)

        inputs.str(
            "athlete_sample_id",
            label="Select Analyzed Athlete Video",
            required=True,
            view=choices,
        )
        inputs.str(
            "problem_timestamp",
            label="Focus on Specific Timestamp (optional)",
            description="e.g. 0:04 — leave empty to analyze all problems",
            required=False,
        )

        return types.Property(inputs)

    def execute(self, ctx):
        from .countervision import find_correct_form

        sample_id = ctx.params.get("athlete_sample_id")
        if not sample_id:
            return {"status": "error", "matches_found": 0, "details": "No sample selected"}

        ds = fo.load_dataset("coachme-athletes")
        sample = ds[sample_id]
        sport = _get_sample_field(sample, "sport", "general")
        index_id = _get_sample_field(sample, "tl_index_id", "")

        if not index_id:
            return {"status": "error", "matches_found": 0, "details": "No Twelve Labs index ID on sample"}

        client = get_client()

        ctx.set_progress(label="Running CounterVision analysis...", progress=0.1)
        matches = find_correct_form(sample, index_id, client, ctx=ctx)

        # Write results back to sample
        sample["counterfactual_matches"] = matches
        sample.save()

        ctx.set_progress(label="CounterVision complete!", progress=1.0)
        ctx.trigger("open_dataset", {"name": "coachme-athletes"})

        details = "\n".join(
            f"[{m.get('timestamp', '?')}] {m.get('problem_description', '?')} → confidence: {m.get('confidence', 0)}%"
            for m in matches
        )

        return {
            "status": "done",
            "matches_found": len(matches),
            "details": details or "No problems identified to match",
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("status", label="Status")
        outputs.int("matches_found", label="Correct-Form Matches Found")
        outputs.str("details", label="Match Details")
        return types.Property(outputs)


# ─────────────────────────────────────────────
# Operator 5: Sync Technique (TechniqueSync)
# ─────────────────────────────────────────────

class SyncTechnique(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="sync_technique",
            label="CoachMe+: TechniqueSync — Phase Alignment",
            description="Temporally align athlete vs reference video by movement phases",
            icon="/assets/icon.svg",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        # Athlete sample dropdown
        if not fo.dataset_exists("coachme-athletes"):
            inputs.view("no_data", types.Warning(label="No athlete analyses found. Run 'Analyze My Technique' first."))
            return types.Property(inputs)

        ds = fo.load_dataset("coachme-athletes")
        if len(ds) == 0:
            inputs.view("empty", types.Warning(label="No analyzed videos."))
            return types.Property(inputs)

        athlete_choices = types.DropdownView()
        for sample in ds:
            sport = _get_sample_field(sample, "sport", "?")
            label = f"{sport.upper()} | {os.path.basename(sample.filepath)}"
            athlete_choices.add_choice(sample.id, label=label)

        inputs.str(
            "athlete_sample_id",
            label="Select Athlete Video",
            required=True,
            view=athlete_choices,
        )

        # Reference sample dropdown — populate based on athlete's sport
        athlete_id = ctx.params.get("athlete_sample_id")
        if athlete_id:
            try:
                athlete_sample = ds[athlete_id]
                sport = _get_sample_field(athlete_sample, "sport", "general")
                ref_ds_name = f"coachme-reference-{sport}"

                if fo.dataset_exists(ref_ds_name):
                    ref_ds = fo.load_dataset(ref_ds_name)
                    ref_choices = types.DropdownView()
                    for ref_sample in ref_ds:
                        ref_choices.add_choice(ref_sample.id, label=os.path.basename(ref_sample.filepath))
                    inputs.str(
                        "reference_sample_id",
                        label="Select Reference Video to Compare Against",
                        required=True,
                        view=ref_choices,
                    )
                else:
                    inputs.view(
                        "no_refs",
                        types.Warning(label=f"No reference dataset found for '{sport}'."),
                    )
            except Exception:
                pass

        inputs.str(
            "skill_name",
            label="Skill / Movement Name",
            description="e.g. jab, squat, serve, deadlift",
            required=True,
        )

        return types.Property(inputs)

    def execute(self, ctx):
        from .technique_sync import sync_technique

        athlete_id = ctx.params.get("athlete_sample_id")
        reference_id = ctx.params.get("reference_sample_id")
        skill_name = ctx.params.get("skill_name", "technique")

        if not athlete_id or not reference_id:
            return {"status": "error", "overall_sync_score": 0.0, "phases_aligned": 0}

        ds = fo.load_dataset("coachme-athletes")
        athlete_sample = ds[athlete_id]
        sport = _get_sample_field(athlete_sample, "sport", "general")

        ref_ds = fo.load_dataset(f"coachme-reference-{sport}")
        reference_sample = ref_ds[reference_id]

        client = get_client()

        ctx.set_progress(label="Running TechniqueSync...", progress=0.1)
        alignment = sync_technique(athlete_sample, reference_sample, skill_name, client, ctx=ctx)

        # Write results back to athlete sample
        athlete_sample["phase_alignment"] = alignment
        athlete_sample.save()

        ctx.set_progress(label="TechniqueSync complete!", progress=1.0)
        ctx.trigger("open_dataset", {"name": "coachme-athletes"})

        return {
            "status": "done",
            "overall_sync_score": alignment.get("overall_sync_score", 0.0),
            "phases_aligned": len(alignment.get("phases", [])),
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("status", label="Status")
        outputs.float("overall_sync_score", label="Overall Sync Score")
        outputs.int("phases_aligned", label="Phases Aligned")
        return types.Property(outputs)


# ─────────────────────────────────────────────
# Operator 6: Validate Coaching (CoachCheck)
# ─────────────────────────────────────────────

class ValidateCoaching(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="validate_coaching",
            label="CoachMe+: CoachCheck — Validate AI Feedback",
            description="Detect hallucinations in AI coaching feedback by cross-verifying each claim",
            icon="/assets/icon.svg",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        if not fo.dataset_exists("coachme-athletes"):
            inputs.view("no_data", types.Warning(label="No athlete analyses found. Run 'Analyze My Technique' first."))
            return types.Property(inputs)

        ds = fo.load_dataset("coachme-athletes")
        if len(ds) == 0:
            inputs.view("empty", types.Warning(label="No analyzed videos."))
            return types.Property(inputs)

        choices = types.DropdownView()
        for sample in ds:
            sport = _get_sample_field(sample, "sport", "?")
            score = _get_sample_field(sample, "similarity_score", 0)
            label = f"{sport.upper()} | Score: {score}% | {os.path.basename(sample.filepath)}"
            choices.add_choice(sample.id, label=label)

        inputs.str(
            "athlete_sample_id",
            label="Select Analyzed Athlete Video",
            description="Pick a video that has coaching feedback to validate",
            required=True,
            view=choices,
        )

        return types.Property(inputs)

    def execute(self, ctx):
        from .coach_check import validate_coaching

        sample_id = ctx.params.get("athlete_sample_id")
        if not sample_id:
            return {"status": "error", "grounding_score": 0.0, "hallucinations_flagged": 0, "details": "No sample selected"}

        ds = fo.load_dataset("coachme-athletes")
        sample = ds[sample_id]

        client = get_client()

        ctx.set_progress(label="Running CoachCheck validation...", progress=0.1)
        validation = validate_coaching(sample, client, ctx=ctx)

        # Write results back to sample
        sample["coaching_validation"] = validation
        sample.save()

        ctx.set_progress(label="CoachCheck complete!", progress=1.0)
        ctx.trigger("open_dataset", {"name": "coachme-athletes"})

        details_lines = []
        for claim in validation.get("claims", []):
            status = "GROUNDED" if claim.get("is_grounded") else "HALLUCINATION"
            details_lines.append(
                f"[{status}] [{claim.get('timestamp', '?')}] {claim.get('original_claim', '?')}"
            )

        return {
            "status": "done",
            "grounding_score": validation.get("grounding_score", 0.0),
            "hallucinations_flagged": validation.get("hallucinations_flagged", 0),
            "details": "\n".join(details_lines) or "No claims to validate",
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("status", label="Status")
        outputs.float("grounding_score", label="Grounding Score (%)")
        outputs.int("hallucinations_flagged", label="Hallucinations Flagged")
        outputs.str("details", label="Claim Validation Details")
        return types.Property(outputs)


# ─────────────────────────────────────────────
# Operator 7: Profile References (TrainingDNA)
# ─────────────────────────────────────────────

class ProfileReferences(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="profile_references",
            label="CoachMe+: TrainingDNA — Profile Reference Library",
            description="Generate a health card for your reference video library — coverage, gaps, and duplicates",
            icon="/assets/icon.svg",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str(
            "sport",
            label="Sport to Profile",
            description="Must match the sport name used when loading references",
            required=True,
        )
        inputs.float(
            "similarity_threshold",
            label="Near-Duplicate Threshold (%)",
            description="Videos above this similarity % are flagged as near-duplicates (default: 92)",
            required=False,
            default=92.0,
        )

        sport = ctx.params.get("sport", "")
        if sport:
            ds_name = f"coachme-reference-{sport.strip().lower()}"
            if fo.dataset_exists(ds_name):
                ds = fo.load_dataset(ds_name)
                inputs.view(
                    "preview",
                    types.Notice(label=f"Found dataset '{ds_name}' with {len(ds)} video(s)"),
                )
            else:
                inputs.view(
                    "not_found",
                    types.Warning(label=f"No reference dataset found for '{sport}'. Load reference videos first."),
                )

        return types.Property(inputs)

    def execute(self, ctx):
        from .training_dna import profile_references

        sport = ctx.params.get("sport", "").strip().lower()
        threshold_pct = ctx.params.get("similarity_threshold", 92.0)
        threshold = threshold_pct / 100.0  # Convert % to 0-1

        ds_name = f"coachme-reference-{sport}"
        if not fo.dataset_exists(ds_name):
            return {
                "status": "error",
                "total_videos": 0,
                "coverage_gaps": f"No reference dataset for '{sport}'",
                "recommendations": "Run 'Load Reference Videos' first",
            }

        dataset = fo.load_dataset(ds_name)
        client = get_client()

        ctx.set_progress(label="Running TrainingDNA analysis...", progress=0.1)
        profile = profile_references(dataset, sport, client, threshold, ctx=ctx)

        # Store profile as dataset info (metadata)
        dataset.info["training_dna"] = profile
        dataset.save()

        ctx.set_progress(label="TrainingDNA complete!", progress=1.0)
        ctx.trigger("open_dataset", {"name": ds_name})

        return {
            "status": "done",
            "total_videos": profile.get("total_videos", 0),
            "coverage_gaps": "\n".join(profile.get("coverage_gaps", [])) or "None found",
            "recommendations": "\n".join(profile.get("recommendations", [])) or "Library looks good",
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("status", label="Status")
        outputs.int("total_videos", label="Total Reference Videos")
        outputs.str("coverage_gaps", label="Coverage Gaps")
        outputs.str("recommendations", label="Recommendations")
        return types.Property(outputs)


# ─────────────────────────────────────────────
# Register all 7 operators
# ─────────────────────────────────────────────

def register(p):
    p.register(LoadReferenceVideos)
    p.register(AnalyzeTechnique)
    p.register(ViewCoachingReport)
    p.register(FindCorrectForm)
    p.register(SyncTechnique)
    p.register(ValidateCoaching)
    p.register(ProfileReferences)
