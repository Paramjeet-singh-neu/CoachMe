# CoachMe+ — AI Sports Coach with Hallucination Detection & Dataset Profiling

> "Elite athletes have coaches. Everyone else has CoachMe+."

A **FiftyOne plugin** that uses **Twelve Labs** video understanding to analyze sports technique, generate timestamped AI coaching feedback, auto-find correct form references, temporally align athlete vs pro performance, **validate its own AI analysis**, and **profile your reference library for gaps**.

Built for the [Voxel51 + Twelve Labs Video Understanding Hackathon](https://voxel51.com/events/video-understanding-ai-hackathon-at-northeastern-april-3-2026) at Northeastern University, April 3, 2026.

**Team:** Aatmaj Rajore, Paramjeet Singh & Tanay Mihani

---

## The Problem

99% of athletes train without qualified coaching. Even the best AI coaching tools stop at "here's what's wrong." They never answer:

- **"Can I trust this AI feedback?"** — Is the coaching grounded in the video, or is the model hallucinating?
- **"Is my reference library good enough?"** — What techniques are covered, what's missing, what's redundant?

CoachMe+ answers both.

## Features

| # | Operator | Module | Twelve Labs Model | Purpose |
|---|----------|--------|-------------------|---------|
| 1 | `load_reference_videos` | Core | Marengo | Index pro/coach reference videos |
| 2 | `analyze_technique` | Core | Marengo + Pegasus | Score + coaching feedback |
| 3 | `view_coaching_report` | Core | — | Browse past analyses |
| 4 | `find_correct_form` | CounterVision | Pegasus + Marengo | Auto-find correct form for each problem |
| 5 | `sync_technique` | TechniqueSync | Pegasus + Marengo | Temporal phase alignment |
| 6 | `validate_coaching` | **CoachCheck** | Pegasus | Hallucination detection on AI coaching |
| 7 | `profile_references` | **TrainingDNA** | Pegasus + Marengo | Reference library health card |

## CoachCheck — AI Coaching Hallucination Detector

Validates that AI coaching feedback is actually grounded in the video. Extracts each specific claim, then independently asks Pegasus to verify it against the actual video at the referenced timestamp.

### How it works

1. Extract factual claims from coaching feedback (timestamps + body parts)
2. For each claim, ask Pegasus independently: "What do you see at this timestamp?"
3. Compare verification against the original claim
4. Flag ungrounded claims as hallucinations

### Real test results

**Test 1 — Legitimate coaching (QEVD exercise video):**
```
Grounding Score: 100.0%
Claims checked: 5
Hallucinations: 0

1. [GROUNDED] "The woman begins with feet together and arms at sides"
2. [GROUNDED] "As she lifts her right leg, upper body remains stable"
3. [GROUNDED] "Arms extend to sides, posture remains upright"
4. [GROUNDED] "In plank position, body forms straight line head to heels"
5. [GROUNDED] "Return to starting position with deliberate control"
```

**Test 2 — Injected hallucinations (3 real + 3 fake claims):**
```
Grounding Score: 50.0%
Claims checked: 6
Hallucinations flagged: 3

1. [GROUNDED]      "Left boxer executes a jab with lead hand"
2. [HALLUCINATION] "Boxer performs a spinning back kick, rotating 360 degrees"
3. [GROUNDED]      "Right boxer shifts weight onto back foot during punch"
4. [HALLUCINATION] "Left boxer drops to ground and performs a leg sweep"
5. [GROUNDED]      "Left boxer maintains high guard, hands close to face"
6. [HALLUCINATION] "Both fighters pull out nunchucks and begin weapons exchange"
```

**6/6 correct classifications. All hallucinations caught.**

## TrainingDNA — Reference Library Health Card

Profiles your reference video collection before coaching begins. Detects what techniques are covered, what's missing, camera angle distribution, skill level gaps, and near-duplicate clips.

### How it works

1. Pegasus describes each reference video (technique, angle, skill level, phases)
2. Normalize verbose AI output into clean categories
3. Compute distributions and detect gaps against known technique taxonomy
4. Find near-duplicates via Marengo search or text similarity
5. Generate actionable recommendations

### Real test results (15 QEVD exercise videos)

```
Total Videos:  15

Technique Distribution:
  stretching:        4
  leg stretches:     2
  balance:           2 (+ stretching combos)
  boxing:            1
  squat:             1
  jumping jacks:     1
  leg lifts:         2
  leg kicks:         1
  warm-up:           1

Angle Distribution:
  front: 15  (100%)

Skill Level Distribution:
  intermediate: 15

Near-Duplicates: 24 pairs detected

Coverage Gaps:
  - Only one camera angle (front). Add side/overhead angles.
  - No professional-level references.

Recommendations:
  - Add side and overhead angles for rotation and depth analysis
  - Add pro footage for stronger coaching baseline
  - Add beginner-level references for common mistake contrast
  - 24 near-duplicate pairs — consider removing redundant clips
  - Imbalanced: 'stretching' has 4 clips vs 'boxing' with 1
```

## Architecture

```
                    ┌──────────────────────────────────┐
                    │         FiftyOne App              │
                    │     (Browse + Visualize)          │
                    └──────────────┬───────────────────┘
                                   │
                    ┌──────────────▼───────────────────┐
                    │        CoachMe+ Plugin            │
                    │                                   │
                    │  CORE          │  COUNTERVISION   │
                    │  TECHNIQUESYNC │  COACHCHECK      │
                    │  TRAININGDNA                      │
                    └──────────────┬───────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
    ┌─────────▼──┐      ┌─────────▼─────┐     ┌────────▼───────┐
    │  Marengo   │      │   Pegasus     │     │   FiftyOne     │
    │  3.0       │      │   1.2         │     │   Dataset      │
    │            │      │               │     │                │
    │ Embeddings │      │ Coaching      │     │ Persistent     │
    │ Similarity │      │ Verification  │     │ Storage        │
    │ Search     │      │ Description   │     │                │
    └────────────┘      └───────────────┘     └────────────────┘
       Twelve Labs API                           Local
```

## Installation

### Prerequisites

- Python 3.9-3.12
- Twelve Labs API key ([free signup](https://twelvelabs.io))
- FFmpeg installed

### Setup

```bash
# Clone
git clone https://github.com/Paramjeet-singh-neu/CoachMe.git
cd CoachMe

# Environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# API key
cp .env.example .env
# Edit .env and add your TWELVELABS_API_KEY

# Register plugin with FiftyOne
ln -s $(pwd)/coachme $(python -c "import fiftyone; print(fiftyone.config.plugins_dir)")/coachme

# Verify
python -c "import fiftyone.plugins as fop; [print(p.name, p.operators) for p in fop.list_plugins()]"
```

## Usage

### Via FiftyOne App

```bash
fiftyone app launch
```

Press `` ` `` (backtick) to open the operator browser. Search "CoachMe" to see all operators.

### Via Python

```python
import fiftyone as fo
import fiftyone.operators as foo

# Load the hackathon exercise dataset
from fiftyone.utils.huggingface import load_from_hub
dataset = load_from_hub("Voxel51/qualcomm-exercise-video-dataset-benchmark")

# Profile your reference library (TrainingDNA)
foo.execute_operator("@team/coachme/profile_references", sport="exercise")

# After running analyze_technique (Core), validate the coaching
foo.execute_operator("@team/coachme/validate_coaching", athlete_sample_id="<sample_id>")
```

## Dataset

We use the **QEVD Benchmark** (Qualcomm Exercise Video Dataset) — 74 workout sessions with temporal annotations for AI fitness coaching, loaded directly from HuggingFace:

```python
from fiftyone.utils.huggingface import load_from_hub
dataset = load_from_hub("Voxel51/qualcomm-exercise-video-dataset-benchmark", max_samples=15)
```

## CV4Smalls Alignment

This project embodies the CV4Smalls philosophy — **data curation beats model complexity**:

- **TrainingDNA** profiles your data before coaching begins — curation-first approach
- **CoachCheck** validates AI output quality — ensuring trustworthy results
- No custom model training — smart use of existing Marengo + Pegasus foundation models
- Reference library gap detection ensures coaching quality through better data

## Tech Stack

| Component | Technology |
|-----------|------------|
| Dataset management + Plugin framework | FiftyOne |
| Video embeddings + similarity + search | Twelve Labs Marengo 3.0 |
| Video-to-text generation + verification | Twelve Labs Pegasus 1.2 |
| Numerical computation | NumPy |
| Runtime | Python 3.12 |

## Project Structure

```
CoachMe/
├── coachme/
│   ├── __init__.py          ← All plugin operators + registration
│   ├── fiftyone.yml         ← Plugin metadata
│   ├── coach_check.py       ← CoachCheck hallucination detection
│   └── training_dna.py      ← TrainingDNA dataset profiling
├── demo-videos/
│   ├── reference/           ← Pro technique videos
│   └── athlete/             ← Test athlete videos
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## License

MIT
