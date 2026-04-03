"""
CoachMe+ FiftyOne Plugin
========================
AI Sports Coach with Counterfactual Analysis, Temporal Alignment,
Hallucination Detection & Dataset Profiling.

Operators defined here (Tanay's modules):
    - validate_coaching   (CoachCheck)
    - profile_references  (TrainingDNA)

Paramjeet will add:
    - load_reference_videos  (Core)
    - analyze_technique      (Core)
    - view_coaching_report   (Core)

Aatmaj will add:
    - find_correct_form   (CounterVision)
    - sync_technique      (TechniqueSync)
"""

import os

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types

from twelvelabs import TwelveLabs
from dotenv import load_dotenv

from .coach_check import validate_coaching
from .training_dna import profile_references

load_dotenv()


def _get_client(ctx):
    """Get an initialized TwelveLabs client.

    Tries ctx.secret first (FiftyOne secrets), then env var.
    """
    api_key = None
    try:
        api_key = ctx.secret("TWELVELABS_API_KEY")
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("TWELVELABS_API_KEY")
    if not api_key:
        raise ValueError(
            "TWELVELABS_API_KEY not found. Set it in .env or FiftyOne secrets."
        )
    return TwelveLabs(api_key=api_key)


# ======================================================================
#  CoachCheck — Hallucination Detection
# ======================================================================

class ValidateCoaching(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="validate_coaching",
            label="CoachCheck — Validate Coaching Feedback",
            description=(
                "Verify that AI coaching claims are grounded in the video. "
                "Detects hallucinated observations."
            ),
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        # Let user pick a sample that already has coaching feedback
        inputs.str(
            "athlete_sample_id",
            label="Athlete Sample ID",
            description=(
                "ID of a previously analyzed athlete sample "
                "(must have coaching_feedback and tl_video_id fields)"
            ),
            required=True,
        )

        return types.Property(
            inputs,
            view=types.View(label="CoachCheck — Validate Coaching"),
        )

    def execute(self, ctx):
        sample_id = ctx.params["athlete_sample_id"]
        dataset = ctx.dataset

        # Load the sample
        sample = dataset[sample_id]

        # Validate required fields exist
        coaching = sample.get("coaching_feedback")
        video_id = sample.get("tl_video_id")
        if not coaching or not video_id:
            return {
                "error": (
                    "Sample is missing 'coaching_feedback' or 'tl_video_id'. "
                    "Run analyze_technique first."
                )
            }

        ctx.set_progress(0.1, "Initializing Twelve Labs client...")
        client = _get_client(ctx)

        ctx.set_progress(0.2, "Extracting claims from coaching feedback...")

        # Run validation
        result = validate_coaching(sample, client)

        # Save results back to sample
        ctx.set_progress(0.8, "Saving validation results...")
        sample["coaching_validation"] = result
        sample.save()

        ctx.set_progress(1.0, "CoachCheck complete!")

        # Summary for the operator output panel
        grounded = result["grounding_score"]
        flagged = result["hallucinations_flagged"]
        total = len(result["claims"])

        return {
            "status": "success",
            "total_claims": total,
            "grounded_claims": total - flagged,
            "hallucinations_flagged": flagged,
            "grounding_score": f"{grounded}%",
            "message": (
                f"Verified {total} coaching claims. "
                f"{total - flagged} grounded, {flagged} flagged as hallucinations. "
                f"Grounding score: {grounded}%"
            ),
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("status", label="Status")
        outputs.int("total_claims", label="Total Claims Checked")
        outputs.int("grounded_claims", label="Grounded Claims")
        outputs.int("hallucinations_flagged", label="Hallucinations Flagged")
        outputs.str("grounding_score", label="Grounding Score")
        outputs.str("message", label="Summary")
        return types.Property(outputs)


# ======================================================================
#  TrainingDNA — Reference Library Health Card
# ======================================================================

class ProfileReferences(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="profile_references",
            label="TrainingDNA — Profile Reference Library",
            description=(
                "Generate a health card for your reference video library. "
                "Shows technique coverage, angle distribution, gaps, "
                "and near-duplicate clips."
            ),
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        inputs.str(
            "sport",
            label="Sport",
            description="Sport name (e.g. boxing, tennis, weightlifting)",
            required=True,
        )

        inputs.str(
            "index_id",
            label="Twelve Labs Index ID (optional)",
            description=(
                "If provided, uses Marengo search for duplicate detection. "
                "Otherwise falls back to text-based similarity."
            ),
            required=False,
        )

        inputs.float(
            "similarity_threshold",
            label="Duplicate Similarity Threshold",
            description="Jaccard threshold for text-based duplicate detection (0-1)",
            default=0.6,
            required=False,
        )

        return types.Property(
            inputs,
            view=types.View(label="TrainingDNA — Profile References"),
        )

    def execute(self, ctx):
        sport = ctx.params["sport"]
        index_id = ctx.params.get("index_id") or None
        threshold = ctx.params.get("similarity_threshold", 0.6)

        dataset = ctx.dataset
        if not dataset or len(dataset) == 0:
            return {"error": "Dataset is empty. Load reference videos first."}

        ctx.set_progress(0.1, "Initializing Twelve Labs client...")
        client = _get_client(ctx)

        ctx.set_progress(0.2, f"Profiling {len(dataset)} reference videos...")

        # Run profiling
        result = profile_references(
            dataset=dataset,
            sport=sport,
            client=client,
            index_id=index_id,
            similarity_threshold=threshold,
        )

        # Store as dataset-level info (metadata)
        ctx.set_progress(0.9, "Saving TrainingDNA profile...")
        dataset.info["training_dna"] = result
        dataset.save()

        ctx.set_progress(1.0, "TrainingDNA profile complete!")

        # Build summary
        tech_summary = ", ".join(
            f"{k}: {v}" for k, v in result["technique_distribution"].items()
        ) or "none detected"
        angle_summary = ", ".join(
            f"{k}: {v}" for k, v in result["angle_distribution"].items()
        ) or "none detected"
        gaps_count = len(result["coverage_gaps"])
        dupes_count = len(result["near_duplicates"])
        recs_count = len(result["recommendations"])

        return {
            "status": "success",
            "total_videos": result["total_videos"],
            "sport": sport,
            "technique_distribution": tech_summary,
            "angle_distribution": angle_summary,
            "coverage_gaps": gaps_count,
            "near_duplicates": dupes_count,
            "recommendations": recs_count,
            "message": (
                f"Profiled {result['total_videos']} {sport} reference videos. "
                f"Techniques: {tech_summary}. "
                f"Gaps: {gaps_count}. Duplicates: {dupes_count}. "
                f"Recommendations: {recs_count}."
            ),
            "full_profile": result,
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("status", label="Status")
        outputs.int("total_videos", label="Total Videos")
        outputs.str("sport", label="Sport")
        outputs.str("technique_distribution", label="Techniques")
        outputs.str("angle_distribution", label="Angles")
        outputs.int("coverage_gaps", label="Coverage Gaps")
        outputs.int("near_duplicates", label="Near-Duplicates")
        outputs.int("recommendations", label="Recommendations")
        outputs.str("message", label="Summary")
        return types.Property(outputs)


# ======================================================================
#  Plugin Registration
# ======================================================================

def register(p):
    """Register all CoachMe+ operators with FiftyOne.

    Called automatically by FiftyOne's plugin loader.
    """
    # ── Tanay's modules ──
    p.register(ValidateCoaching)
    p.register(ProfileReferences)

    # ── Paramjeet: add Core operators here ──
    # p.register(LoadReferenceVideos)
    # p.register(AnalyzeTechnique)
    # p.register(ViewCoachingReport)

    # ── Aatmaj: add add-on operators here ──
    # p.register(FindCorrectForm)
    # p.register(SyncTechnique)
