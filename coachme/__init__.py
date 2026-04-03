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

# Load .env from the project directory (not the plugin install location)
_plugin_dir = os.path.dirname(os.path.abspath(__file__))
_project_env = os.path.join(os.path.dirname(_plugin_dir), ".env")
load_dotenv(_project_env)
load_dotenv()  # Also try default locations


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

                # ── HEADER: Overview Dashboard ──
                score_label = "Excellent" if score >= 85 else "Good" if score >= 70 else "Needs Work"
                header_md = (
                    f"# CoachMe+ Report: {sport.upper()}\n\n"
                    f"| Metric | Value |\n"
                    f"|--------|-------|\n"
                    f"| **Similarity Score** | **{score}%** ({score_label}) |\n"
                    f"| **Focus Area** | {focus} |\n"
                    f"| **Analyzed** | {analyzed} |\n"
                    f"| **Video** | {os.path.basename(sample.filepath)} |\n"
                )
                inputs.view("header", types.MarkdownView(label=header_md))

                # ── SECTION 1: Reference Matches ──
                if matches:
                    match_md = "## Reference Matches\n\n"
                    match_md += "| Reference Video | Similarity |\n|---|---|\n"
                    for m in matches:
                        pct = m.get("similarity_pct", 0)
                        bar = ">" * int(pct / 5) if pct else ""
                        match_md += f"| {os.path.basename(m.get('reference_filepath', '?'))} | **{pct}%** {bar} |\n"
                    inputs.view("matches_info", types.MarkdownView(label=match_md))

                # ── SECTION 2: AI Coaching Feedback ──
                feedback_md = "## AI Coaching Feedback (Pegasus)\n\n" + feedback
                inputs.view("feedback_display", types.MarkdownView(label=feedback_md))

                # ── SECTION 3: CoachCheck Validation ──
                if validation:
                    g_score = validation.get("grounding_score", 0)
                    h_count = validation.get("hallucinations_flagged", 0)
                    claims = validation.get("claims", [])
                    total_claims = len(claims)
                    trust_label = "High Trust" if g_score >= 80 else "Moderate Trust" if g_score >= 60 else "Low Trust"

                    val_md = (
                        f"## CoachCheck: Hallucination Detection\n\n"
                        f"**Grounding Score: {g_score}%** ({trust_label}) | "
                        f"**{total_claims - h_count}/{total_claims}** claims verified | "
                        f"**{h_count}** hallucination(s) caught\n\n"
                    )
                    if claims:
                        val_md += "| Status | Claim | Timestamp | Confidence |\n|---|---|---|---|\n"
                        for c in claims:
                            status = "GROUNDED" if c.get("is_grounded") else "HALLUCINATION"
                            val_md += (
                                f"| **{status}** | {c.get('original_claim', '?')[:80]} | "
                                f"{c.get('timestamp', '?')} | {c.get('confidence', '?')} |\n"
                            )
                    inputs.view("validation_info", types.MarkdownView(label=val_md))

                # ── SECTION 4: CounterVision ──
                if counterfactuals:
                    cf_md = f"## CounterVision: What Should You Have Done?\n\n"
                    cf_md += f"Found **{len(counterfactuals)}** problem-to-correction matches:\n\n"
                    for i, cf in enumerate(counterfactuals, 1):
                        conf = cf.get("confidence", 0)
                        cf_md += (
                            f"### Problem {i}: [{cf.get('timestamp', '?')}]\n"
                            f"**Issue:** {cf.get('problem_description', '?')}\n\n"
                            f"**Correct Form:** {cf.get('correct_description', '?')}\n\n"
                            f"**Match Confidence:** {conf}%\n\n"
                            f"---\n\n"
                        )
                    inputs.view("counterfactual_info", types.MarkdownView(label=cf_md))

                # ── SECTION 5: TechniqueSync ──
                if phase_align:
                    sync_score = phase_align.get("overall_sync_score", 0)
                    phases = phase_align.get("phases", [])
                    sync_label = "Excellent" if sync_score >= 0.9 else "Good" if sync_score >= 0.7 else "Needs Work"

                    sync_md = (
                        f"## TechniqueSync: Phase Alignment\n\n"
                        f"**Overall Sync Score: {sync_score:.1%}** ({sync_label})\n\n"
                    )
                    if phases:
                        sync_md += "| Phase | Your Timing | Pro Timing | Delta | Alignment |\n|---|---|---|---|---|\n"
                        for p in phases:
                            delta = p.get("timing_delta_seconds", 0)
                            if delta > 0.1:
                                direction = f"+{delta:.2f}s SLOWER"
                            elif delta < -0.1:
                                direction = f"{delta:.2f}s FASTER"
                            else:
                                direction = "ON PACE"
                            a_score = p.get("alignment_score", 0)
                            sync_md += (
                                f"| **{p.get('athlete_phase', '?')}** | {p.get('athlete_time', '?')} | "
                                f"{p.get('reference_time', '?')} | {direction} | {a_score:.1%} |\n"
                            )
                    inputs.view("sync_info", types.MarkdownView(label=sync_md))

                # ── FOOTER: Features used ──
                features_used = ["Core Analysis"]
                if validation: features_used.append("CoachCheck")
                if counterfactuals: features_used.append("CounterVision")
                if phase_align: features_used.append("TechniqueSync")
                footer_md = (
                    f"\n---\n*CoachMe+ features active: {' | '.join(features_used)}*\n\n"
                    f"*\"Elite athletes have coaches. Everyone else has CoachMe+.\"*"
                )
                inputs.view("footer", types.MarkdownView(label=footer_md))

            except Exception as e:
                inputs.view("err", types.Warning(label=f"Could not load this sample: {e}"))

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
        profile = profile_references(dataset, sport, client, similarity_threshold=threshold, ctx=ctx)

        # Store profile as dataset info (metadata)
        dataset.info["training_dna"] = profile

        # ── DATA CURATION: tag each sample with technique, angle, quality ──
        ctx.set_progress(label="Tagging samples with curation metadata...", progress=0.9)
        video_profiles = profile.get("video_descriptions", [])
        for vp in video_profiles:
            filepath = vp.get("filepath", "")
            if not filepath:
                continue
            matched = dataset.match(F("filepath") == filepath)
            if len(matched) == 0:
                continue
            s = matched.first()
            s["technique"] = vp.get("technique", "unknown")
            s["camera_angle"] = vp.get("angle", "unknown")
            s["skill_level"] = vp.get("skill_level", "unknown")
            s["curation_quality"] = vp.get("duration_quality", "unknown")
            s.save()

        # Tag near-duplicates
        for nd in profile.get("near_duplicates", []):
            for fp in [nd.get("video_a", ""), nd.get("video_b", "")]:
                if not fp:
                    continue
                matched = dataset.match(F("filepath") == fp)
                if len(matched) > 0:
                    s = matched.first()
                    s.tags.append("near-duplicate")
                    s.save()

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
# Operator 8: Curate References (Data Curation)
# ─────────────────────────────────────────────

class CurateReferences(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="curate_references",
            label="CoachMe+: Curate Reference Library",
            description="Compute embedding similarity, find duplicates, tag quality — smart data curation for your reference library",
            icon="/assets/icon.svg",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str(
            "sport",
            label="Sport to Curate",
            required=True,
        )

        sport = ctx.params.get("sport", "")
        if sport:
            ds_name = f"coachme-reference-{sport.strip().lower()}"
            if fo.dataset_exists(ds_name):
                ds = fo.load_dataset(ds_name)
                inputs.view(
                    "info",
                    types.Notice(label=f"'{ds_name}' has {len(ds)} video(s). Curation will:\n"
                        f"• Compute pairwise embedding similarity\n"
                        f"• Build similarity index for visual exploration\n"
                        f"• Tag near-duplicates for removal\n"
                        f"• Score each video's uniqueness"),
                )
            else:
                inputs.view("warn", types.Warning(label=f"No dataset for '{sport}'."))

        return types.Property(inputs)

    def execute(self, ctx):
        sport = ctx.params.get("sport", "").strip().lower()
        ds_name = f"coachme-reference-{sport}"
        if not fo.dataset_exists(ds_name):
            return {"status": "error", "details": f"No dataset for '{sport}'"}

        dataset = fo.load_dataset(ds_name)
        client = get_client()

        # Step 1: Ensure all samples have embeddings
        ctx.set_progress(label="Computing embeddings...", progress=0.1)
        samples = list(dataset)
        embeddings = []
        for i, s in enumerate(samples):
            emb = _get_sample_field(s, "tl_embedding")
            if emb:
                embeddings.append(np.array(emb))
            else:
                ctx.set_progress(
                    label=f"Embedding {os.path.basename(s.filepath)} ({i+1}/{len(samples)})...",
                    progress=0.1 + 0.4 * i / len(samples),
                )
                emb = get_video_embedding(client, s.filepath)
                s["tl_embedding"] = emb
                s.save()
                embeddings.append(np.array(emb))

        if len(embeddings) < 2:
            return {"status": "done", "details": "Need at least 2 videos to curate"}

        # Step 2: Compute pairwise similarity and uniqueness scores
        ctx.set_progress(label="Computing pairwise similarity...", progress=0.6)
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    similarity_matrix[i][j] = cosine_similarity(embeddings[i], embeddings[j])

        # Step 3: Tag samples with uniqueness and duplicate info
        ctx.set_progress(label="Tagging samples...", progress=0.8)
        dup_threshold = 0.92
        duplicates_found = 0

        for i, s in enumerate(samples):
            # Uniqueness = 1 - max similarity to any other video
            other_sims = [similarity_matrix[i][j] for j in range(n) if j != i]
            max_sim = max(other_sims) if other_sims else 0
            uniqueness = round((1 - max_sim) * 100, 1)
            s["uniqueness_score"] = uniqueness
            s["max_similarity"] = round(max_sim * 100, 1)

            # Find most similar video
            most_similar_idx = max((j for j in range(n) if j != i), key=lambda j: similarity_matrix[i][j])
            s["most_similar_to"] = os.path.basename(samples[most_similar_idx].filepath)

            # Tag near-duplicates
            if max_sim > dup_threshold:
                if "near-duplicate" not in s.tags:
                    s.tags.append("near-duplicate")
                duplicates_found += 1
            else:
                if "unique" not in s.tags:
                    s.tags.append("unique")

            s.save()

        # Step 4: Store similarity matrix as brain result for FiftyOne visualization
        ctx.set_progress(label="Building similarity index...", progress=0.9)
        try:
            import fiftyone.brain as fob
            emb_array = np.array(embeddings)
            dataset.set_values("embedding", [e.tolist() for e in embeddings])
            fob.compute_similarity(dataset, brain_key="technique_sim", embeddings="embedding")
        except Exception:
            pass  # Brain visualization is optional

        dataset.save()
        ctx.set_progress(label="Curation complete!", progress=1.0)
        ctx.trigger("open_dataset", {"name": ds_name})

        # Build summary
        unique_count = sum(1 for s in samples if "unique" in s.tags)
        avg_uniqueness = round(np.mean([_get_sample_field(s, "uniqueness_score", 0) for s in samples]), 1)

        return {
            "status": "done",
            "details": (
                f"Curated {len(samples)} videos\n"
                f"Unique: {unique_count} | Near-duplicates: {duplicates_found}\n"
                f"Avg uniqueness: {avg_uniqueness}%\n"
                f"Tip: Filter by 'near-duplicate' tag to review redundant clips"
            ),
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("status", label="Status")
        outputs.str("details", label="Curation Results")
        return types.Property(outputs)


# ─────────────────────────────────────────────
# Register all 8 operators
# ─────────────────────────────────────────────

def register(p):
    p.register(LoadReferenceVideos)
    p.register(AnalyzeTechnique)
    p.register(ViewCoachingReport)
    p.register(FindCorrectForm)
    p.register(SyncTechnique)
    p.register(ValidateCoaching)
    p.register(ProfileReferences)
    p.register(CurateReferences)
