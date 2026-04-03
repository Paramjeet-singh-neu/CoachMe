"""
CoachMe+ — AI Sports Coach FiftyOne Plugin

Operators:
  Core:
    - load_reference_videos: Index pro/coach reference videos
    - analyze_technique: Compare athlete video, get score + feedback
    - view_coaching_report: Browse past analyses
  CounterVision:
    - find_correct_form: Auto-find correct form for each problem
  TechniqueSync:
    - sync_technique: Temporal phase alignment
  CoachCheck:
    - validate_coaching: Hallucination detection
  TrainingDNA:
    - profile_references: Reference library health card
"""

import os
import time
import json

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
from twelvelabs import TwelveLabs
from dotenv import load_dotenv

# Import add-on modules
from .countervision import find_correct_form as _find_correct_form
from .technique_sync import sync_technique as _sync_technique

# Try importing Tanay's modules (may not exist yet)
try:
    from .coach_check import validate_coaching as _validate_coaching
except ImportError:
    _validate_coaching = None

try:
    from .training_dna import profile_references as _profile_references
except ImportError:
    _profile_references = None


load_dotenv()


def _get_client():
    """Get TwelveLabs client from env."""
    api_key = os.getenv("TWELVELABS_API_KEY")
    if not api_key:
        raise ValueError("TWELVELABS_API_KEY not set in environment or .env file")
    return TwelveLabs(api_key=api_key)


# ─────────────────────────────────────────────
# CORE OPERATORS (Paramjeet fills these in)
# ─────────────────────────────────────────────

class LoadReferenceVideos(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="load_reference_videos",
            label="CoachMe+: Load Reference Videos",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str("sport", label="Sport", required=True, description="e.g., boxing, tennis, weightlifting")
        inputs.str("videos_dir", label="Video Directory Path", required=True, description="Path to folder containing reference videos")
        inputs.str("index_name", label="Index Name", default="", description="Leave blank to auto-generate from sport name")
        return types.Property(inputs, view=types.View(label="Load Reference Videos"))

    def execute(self, ctx):
        client = _get_client()
        sport = ctx.params["sport"]
        videos_dir = ctx.params["videos_dir"]
        index_name = ctx.params.get("index_name") or f"coachme-ref-{sport}"

        # Create or get index
        try:
            index = client.indexes.create(
                index_name=index_name,
                models=[{
                    "model_name": "marengo3.0",
                    "model_options": ["visual", "audio"],
                }],
            )
            index_id = index._id
        except Exception as e:
            # Index may already exist
            if "already exists" in str(e).lower():
                indexes = client.indexes.list()
                index_id = None
                for idx in indexes:
                    if idx.index_name == index_name:
                        index_id = idx._id
                        break
                if not index_id:
                    return {"error": f"Index exists but could not find it: {e}"}
            else:
                return {"error": str(e)}

        # Create/load FiftyOne dataset
        dataset_name = f"coachme-reference-{sport}"
        if fo.dataset_exists(dataset_name):
            dataset = fo.load_dataset(dataset_name)
        else:
            dataset = fo.Dataset(name=dataset_name, persistent=True)

        # Upload videos
        import glob
        video_files = glob.glob(os.path.join(videos_dir, "*.mp4"))
        video_files += glob.glob(os.path.join(videos_dir, "*.mov"))
        video_files += glob.glob(os.path.join(videos_dir, "*.avi"))

        indexed = 0
        for vf in video_files:
            sample = fo.Sample(filepath=vf)
            sample["sport"] = sport
            sample["role"] = "reference"

            try:
                task = client.tasks.create(
                    index_id=index_id,
                    video_file=open(vf, "rb"),
                )
                sample["tl_video_id"] = task.video_id
                sample["tl_index_id"] = index_id
                indexed += 1
            except Exception as e:
                sample["tl_error"] = str(e)

            dataset.add_sample(sample)

        return {
            "status": "success",
            "index_id": index_id,
            "dataset": dataset_name,
            "videos_indexed": indexed,
            "total_videos": len(video_files),
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("status", label="Status")
        outputs.str("index_id", label="Index ID")
        outputs.str("dataset", label="Dataset Name")
        outputs.int("videos_indexed", label="Videos Indexed")
        outputs.int("total_videos", label="Total Videos Found")
        return types.Property(outputs, view=types.View(label="Reference Videos Loaded"))


class AnalyzeTechnique(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="analyze_technique",
            label="CoachMe+: Analyze Technique",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str("sport", label="Sport", required=True, description="e.g., boxing")
        inputs.str("video_path", label="Athlete Video Path", required=True)
        inputs.str("focus", label="Focus Area", default="overall technique", description="e.g., jab technique, footwork")
        return types.Property(inputs, view=types.View(label="Analyze Athlete Technique"))

    def execute(self, ctx):
        client = _get_client()
        sport = ctx.params["sport"]
        video_path = ctx.params["video_path"]
        focus = ctx.params.get("focus", "overall technique")

        # Find reference index
        ref_dataset_name = f"coachme-reference-{sport}"
        if not fo.dataset_exists(ref_dataset_name):
            return {"error": f"No reference dataset found for {sport}. Run 'Load Reference Videos' first."}

        ref_dataset = fo.load_dataset(ref_dataset_name)
        ref_sample = ref_dataset.first()
        if not ref_sample or not ref_sample.get("tl_index_id"):
            return {"error": "Reference dataset has no indexed videos."}

        index_id = ref_sample["tl_index_id"]

        # Upload athlete video
        try:
            task = client.tasks.create(
                index_id=index_id,
                video_file=open(video_path, "rb"),
            )
            athlete_video_id = task.video_id
        except Exception as e:
            return {"error": f"Failed to upload athlete video: {e}"}

        # Wait for indexing
        time.sleep(5)

        # Search for similarity against reference
        try:
            search_results = client.search.query(
                index_id=index_id,
                search_options=["visual"],
                query_text=f"{sport} {focus} correct technique",
                page_limit=5,
            )

            reference_matches = []
            for item in search_results:
                reference_matches.append({
                    "video_id": item.video_id,
                    "start": item.start,
                    "end": item.end,
                    "score": getattr(item, "score", None) or getattr(item, "rank", 0),
                })
        except Exception as e:
            reference_matches = []

        # Generate coaching feedback with Pegasus
        coaching_prompt = f"""You are an expert {sport} coach analyzing an athlete's {focus}.

Analyze this video and provide structured coaching feedback:

1. TECHNIQUE SCORE: Rate from 1-10
2. STRENGTHS: What the athlete does well (2-3 points with timestamps)
3. AREAS FOR IMPROVEMENT: Specific technique errors with timestamps (2-4 points)
4. DRILLS: 1-2 specific drills to fix the main issues
5. PRIORITY FIX: The single most important thing to fix first

Be specific about body positions, angles, timing. Reference exact moments in the video."""

        try:
            coaching_response = client.analyze(
                video_id=athlete_video_id,
                prompt=coaching_prompt,
                temperature=0.3,
                max_tokens=2000,
            )
            coaching_feedback = coaching_response.data
        except Exception as e:
            coaching_feedback = f"Coaching analysis failed: {e}"

        # Compute similarity score from search results
        if reference_matches:
            similarity_score = max(m.get("score", 0) for m in reference_matches)
        else:
            similarity_score = 0.0

        # Store in FiftyOne
        athlete_dataset_name = f"coachme-athletes"
        if fo.dataset_exists(athlete_dataset_name):
            athlete_dataset = fo.load_dataset(athlete_dataset_name)
        else:
            athlete_dataset = fo.Dataset(name=athlete_dataset_name, persistent=True)

        sample = fo.Sample(filepath=video_path)
        sample["sport"] = sport
        sample["role"] = "athlete"
        sample["focus_area"] = focus
        sample["tl_video_id"] = athlete_video_id
        sample["tl_index_id"] = index_id
        sample["similarity_score"] = similarity_score
        sample["reference_matches"] = reference_matches
        sample["coaching_feedback"] = coaching_feedback
        sample["analyzed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        athlete_dataset.add_sample(sample)

        return {
            "status": "success",
            "sample_id": str(sample.id),
            "similarity_score": similarity_score,
            "coaching_feedback": coaching_feedback[:500] + "..." if len(coaching_feedback) > 500 else coaching_feedback,
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("status", label="Status")
        outputs.str("sample_id", label="Sample ID")
        outputs.float("similarity_score", label="Similarity Score")
        outputs.str("coaching_feedback", label="Coaching Feedback")
        return types.Property(outputs, view=types.View(label="Technique Analysis"))


class ViewCoachingReport(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="view_coaching_report",
            label="CoachMe+: View Coaching Report",
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        return types.Property(inputs, view=types.View(label="View All Coaching Reports"))

    def execute(self, ctx):
        dataset_name = "coachme-athletes"
        if not fo.dataset_exists(dataset_name):
            return {"status": "No analyses found. Run 'Analyze Technique' first."}

        dataset = fo.load_dataset(dataset_name)
        ctx.trigger("set_view", params={"view": dataset.view()})
        return {"status": f"Loaded {len(dataset)} athlete analyses"}


# ─────────────────────────────────────────────
# COUNTERVISION OPERATOR (Aatmaj's module)
# ─────────────────────────────────────────────

class FindCorrectForm(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="find_correct_form",
            label="CoachMe+: Find Correct Form (CounterVision)",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str("athlete_sample_id", label="Athlete Sample ID", required=True,
                    description="ID of a previously analyzed athlete sample")
        inputs.str("problem_timestamp", label="Focus on Timestamp (optional)", default="",
                    description="e.g., 0:04-0:06 — leave blank to process all problems")
        return types.Property(inputs, view=types.View(label="Find Correct Form"))

    def execute(self, ctx):
        client = _get_client()
        sample_id = ctx.params["athlete_sample_id"]

        dataset = fo.load_dataset("coachme-athletes")
        sample = dataset[sample_id]

        if not sample.get("coaching_feedback") or not sample.get("tl_index_id"):
            return {"error": "Sample missing coaching_feedback or index. Run 'Analyze Technique' first."}

        matches = _find_correct_form(
            athlete_sample=sample,
            reference_index_id=sample["tl_index_id"],
            client=client,
            sport=sample.get("sport", "sports"),
        )

        # Save results back to sample
        sample["counterfactual_matches"] = matches
        sample.save()

        return {
            "status": "success",
            "problems_found": len(matches),
            "with_references": sum(1 for m in matches if m.get("reference_video_id")),
            "matches": matches,
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("status", label="Status")
        outputs.int("problems_found", label="Problems Found")
        outputs.int("with_references", label="Reference Matches")
        return types.Property(outputs, view=types.View(label="CounterVision Results"))


# ─────────────────────────────────────────────
# TECHNIQUESYNC OPERATOR (Aatmaj's module)
# ─────────────────────────────────────────────

class SyncTechnique(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="sync_technique",
            label="CoachMe+: Sync Technique (TechniqueSync)",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str("athlete_sample_id", label="Athlete Sample ID", required=True)
        inputs.str("reference_sample_id", label="Reference Sample ID", required=True,
                    description="ID of a reference video sample to compare against")
        inputs.str("skill_name", label="Skill Name", required=True,
                    description="e.g., jab, cross, squat, serve")
        return types.Property(inputs, view=types.View(label="Temporal Phase Alignment"))

    def execute(self, ctx):
        client = _get_client()
        athlete_id = ctx.params["athlete_sample_id"]
        reference_id = ctx.params["reference_sample_id"]
        skill_name = ctx.params["skill_name"]

        # Load samples
        athlete_dataset = fo.load_dataset("coachme-athletes")
        athlete_sample = athlete_dataset[athlete_id]

        # Find reference sample — could be in any reference dataset
        ref_sample = None
        for ds_name in fo.list_datasets():
            if ds_name.startswith("coachme-reference"):
                ds = fo.load_dataset(ds_name)
                try:
                    ref_sample = ds[reference_id]
                    break
                except Exception:
                    continue

        if not ref_sample:
            return {"error": f"Reference sample {reference_id} not found in any coachme-reference dataset"}

        result = _sync_technique(
            athlete_sample=athlete_sample,
            reference_sample=ref_sample,
            skill_name=skill_name,
            client=client,
        )

        if "error" in result:
            return result

        # Save to athlete sample
        athlete_sample["phase_alignment"] = result
        athlete_sample.save()

        # Format summary
        summary_lines = []
        for phase in result.get("phases", []):
            delta = phase["timing_delta_seconds"]
            direction = "slower" if delta > 0 else "faster"
            summary_lines.append(
                f"{phase['athlete_phase']}: {abs(delta):.2f}s {direction} than reference"
            )

        return {
            "status": "success",
            "overall_sync_score": result["overall_sync_score"],
            "phases_aligned": len(result.get("phases", [])),
            "summary": "\n".join(summary_lines),
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("status", label="Status")
        outputs.float("overall_sync_score", label="Overall Sync Score")
        outputs.int("phases_aligned", label="Phases Aligned")
        outputs.str("summary", label="Timing Summary")
        return types.Property(outputs, view=types.View(label="TechniqueSync Results"))


# ─────────────────────────────────────────────
# COACHCHECK OPERATOR (Tanay's module — stub)
# ─────────────────────────────────────────────

class ValidateCoaching(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="validate_coaching",
            label="CoachMe+: Validate Coaching (CoachCheck)",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str("athlete_sample_id", label="Athlete Sample ID", required=True)
        return types.Property(inputs, view=types.View(label="Validate Coaching Feedback"))

    def execute(self, ctx):
        if _validate_coaching is None:
            return {"error": "CoachCheck module not yet available (coach_check.py missing)"}

        client = _get_client()
        sample_id = ctx.params["athlete_sample_id"]
        dataset = fo.load_dataset("coachme-athletes")
        sample = dataset[sample_id]

        result = _validate_coaching(sample, client)
        sample["coaching_validation"] = result
        sample.save()

        return {
            "status": "success",
            "grounding_score": result.get("grounding_score", 0),
            "hallucinations_flagged": result.get("hallucinations_flagged", 0),
        }


# ─────────────────────────────────────────────
# TRAININGDNA OPERATOR (Tanay's module — stub)
# ─────────────────────────────────────────────

class ProfileReferences(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="profile_references",
            label="CoachMe+: Profile References (TrainingDNA)",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str("sport", label="Sport", required=True)
        inputs.float("similarity_threshold", label="Near-Duplicate Threshold", default=0.92)
        return types.Property(inputs, view=types.View(label="Profile Reference Library"))

    def execute(self, ctx):
        if _profile_references is None:
            return {"error": "TrainingDNA module not yet available (training_dna.py missing)"}

        client = _get_client()
        sport = ctx.params["sport"]
        threshold = ctx.params.get("similarity_threshold", 0.92)

        dataset_name = f"coachme-reference-{sport}"
        if not fo.dataset_exists(dataset_name):
            return {"error": f"No reference dataset for {sport}. Load references first."}

        dataset = fo.load_dataset(dataset_name)
        result = _profile_references(dataset, sport, client, threshold)
        dataset.info["training_dna"] = result
        dataset.save()

        return {
            "status": "success",
            "total_videos": result.get("total_videos", 0),
            "coverage_gaps": result.get("coverage_gaps", []),
            "near_duplicates": len(result.get("near_duplicates", [])),
        }


# ─────────────────────────────────────────────
# REGISTER ALL OPERATORS
# ─────────────────────────────────────────────

def register(p):
    p.register(LoadReferenceVideos)
    p.register(AnalyzeTechnique)
    p.register(ViewCoachingReport)
    p.register(FindCorrectForm)
    p.register(SyncTechnique)
    p.register(ValidateCoaching)
    p.register(ProfileReferences)
