# CoachMe+ — AI Sports Coach Plugin for FiftyOne

> **"Elite athletes have coaches. Everyone else has CoachMe+."**

A FiftyOne plugin that turns any video library into an AI coaching system. Upload reference technique videos, record yourself, and get instant feedback — similarity scores, timestamped coaching, hallucination-checked analysis, and smart data curation.

**Built by Paramjeet Singh & Aatmaj Rajore** | Voxel51 + Twelve Labs Hackathon, April 2026

---

## Install in 60 Seconds

```bash
# 1. Clone
git clone https://github.com/Paramjeet-singh-neu/CoachMe.git
cd CoachMe

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Twelve Labs API key
cp .env.example .env
# Edit .env and add your key from https://api.twelvelabs.io

# 4. Register the plugin with FiftyOne
mkdir -p $(python -c "import fiftyone; print(fiftyone.config.plugins_dir)")
ln -s $(pwd)/coachme $(python -c "import fiftyone; print(fiftyone.config.plugins_dir)")/coachme

# 5. Launch
fiftyone app launch
```

Press **`` ` ``** (backtick) in the FiftyOne App and search **"CoachMe"** to see all 8 operators.

---

## What It Does

| Step | Operator | What Happens |
|------|----------|-------------|
| 1 | **Load Reference Videos** | Upload pro technique videos, compute Marengo 3.0 embeddings |
| 2 | **Analyze My Technique** | Upload your video, get similarity score + AI coaching feedback |
| 3 | **View Coaching Report** | Browse all results — scores, feedback, matches, validations |
| 4 | **CoachCheck** | Detect hallucinations in AI coaching by cross-verifying each claim |
| 5 | **CounterVision** | For each problem found, auto-find the reference video showing correct form |
| 6 | **TechniqueSync** | Temporally align your movement phases vs the pro — see where you're slower/faster |
| 7 | **TrainingDNA** | Profile your reference library — coverage gaps, duplicates, quality tags |
| 8 | **Curate References** | Compute pairwise similarity, tag duplicates, score uniqueness per video |

---

## Architecture

```
┌──────────────────────────────────────────────┐
│              FiftyOne App                     │
│   8 Operators  ·  Visual Explorer  ·  Tags   │
└──────────────────┬───────────────────────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
┌───▼────┐  ┌─────▼─────┐  ┌────▼─────┐
│Marengo │  │  Pegasus  │  │ Marengo  │
│  3.0   │  │   1.2     │  │  Embed   │
│        │  │           │  │  API     │
│Search  │  │ Coaching  │  │ 512-dim  │
│& Index │  │ & Verify  │  │ Vectors  │
└────────┘  └───────────┘  └──────────┘
         Twelve Labs API
```

**Similarity is real** — we compute 512-dimensional Marengo embeddings per video and use cosine similarity, not text-search proxies.

---

## The 8 Operators

### 1. Load Reference Videos
Index reference technique videos as the coaching baseline.

| Input | Description |
|-------|-------------|
| `sport` | Sport name (boxing, tennis, weightlifting, etc.) |
| `videos_dir` | Path to folder of reference videos (.mp4, .mov, .avi) |

Creates dataset `coachme-reference-{sport}` with embeddings stored per sample.

### 2. Analyze My Technique
Compare your video against references and get AI coaching.

| Input | Description |
|-------|-------------|
| `sport` | Must match sport used for references |
| `video_path` | Path to your video |
| `focus` | Optional focus area (e.g. "elbow position", "footwork") |

**Output:** Similarity % per reference, technique score out of 10, timestamped strengths/weaknesses, drill prescriptions.

### 3. View Coaching Report
Browse all past analyses with a dropdown selector. Displays coaching feedback, similarity scores, and results from CoachCheck, CounterVision, and TechniqueSync when available.

### 4. CoachCheck — Validate AI Feedback
Extracts each factual claim from the coaching feedback, then independently verifies it against the video. Flags hallucinations.

**Output:** Grounding score (%), verified/hallucinated claims with timestamps.

### 5. CounterVision — Find Correct Form
For each problem identified in coaching, searches the reference library for a video segment showing the correct technique.

**Output:** Problem-to-correction matches with timestamps and confidence scores.

### 6. TechniqueSync — Phase Alignment
Breaks both athlete and reference videos into movement phases (stance, extension, recovery, etc.) and aligns them temporally.

**Output:** Per-phase timing deltas (e.g. "your extension is 0.3s slower"), overall sync score.

### 7. TrainingDNA — Profile Reference Library
Generates a health card for your reference collection: technique coverage, camera angles, skill levels, near-duplicates, and gap recommendations. Tags each sample with metadata.

### 8. Curate Reference Library
Computes pairwise embedding similarity across all reference videos. Tags duplicates, scores uniqueness, and builds a FiftyOne similarity index for visual exploration.

**Output:** Uniqueness score per video, near-duplicate tags, most-similar-to links.

---

## Data Fields Written

### Athlete samples (`coachme-athletes`)

| Field | Type | Example |
|-------|------|---------|
| `sport` | str | `"boxing"` |
| `focus_area` | str | `"jab technique"` |
| `tl_video_id` | str | Twelve Labs video ID |
| `similarity_score` | float | `68.1` |
| `reference_matches` | list | `[{video_id, filepath, similarity_pct}]` |
| `coaching_feedback` | str | Timestamped coaching text |
| `coaching_validation` | dict | CoachCheck results |
| `counterfactual_matches` | list | CounterVision results |
| `phase_alignment` | dict | TechniqueSync results |

### Reference samples (`coachme-reference-{sport}`)

| Field | Type | Example |
|-------|------|---------|
| `sport` | str | `"boxing"` |
| `tl_video_id` | str | Twelve Labs video ID |
| `tl_embedding` | list | 512-dim Marengo vector |
| `technique` | str | `"jab"` (after TrainingDNA) |
| `camera_angle` | str | `"front"` (after TrainingDNA) |
| `uniqueness_score` | float | `27.4` (after Curate) |
| `max_similarity` | float | `72.6` (after Curate) |

---

## CLI Demo

```bash
# Core analysis
python demo.py --sport boxing \
    --refs ./demo-videos/reference \
    --athlete ./demo-videos/athlete/My_Video.mp4

# Full pipeline with all features
python demo.py --sport boxing \
    --refs ./demo-videos/reference \
    --athlete ./demo-videos/athlete/My_Video.mp4 \
    --focus "jab technique" \
    --countervision --validate --sync --profile
```

---

## Project Structure

```
CoachMe/
├── coachme/
│   ├── __init__.py          ← All 8 FiftyOne operators
│   ├── fiftyone.yml         ← Plugin metadata
│   ├── countervision.py     ← CounterVision module
│   ├── coach_check.py       ← CoachCheck module
│   ├── technique_sync.py    ← TechniqueSync module
│   ├── training_dna.py      ← TrainingDNA module
│   └── assets/icon.svg      ← Plugin icon
├── demo.py                  ← CLI demo script
├── notebook.ipynb           ← Jupyter demo notebook
├── requirements.txt         ← Dependencies
├── .env.example             ← API key template
└── README.md
```

---

## Supported Sports

Works with any sport that has reference footage. Tested with:
- Boxing (jab, cross, hook, combinations)
- Weightlifting (squat, deadlift)
- Tennis (serve, forehand)
- Cricket (bowling, batting)
- Yoga (poses, sun salutation)
- Swimming (freestyle, backstroke)

---

## Why Data Curation Matters

CoachMe+ isn't just an analysis tool — it's a **data curation system** for coaching video libraries.

| Without Curation | With CoachMe+ Curation |
|------------------|----------------------|
| 50 random YouTube clips | 5 curated clips covering all techniques |
| Duplicate angles everywhere | Uniqueness-scored, duplicates tagged |
| No idea what's missing | Gap analysis: "Missing: hook, uppercut" |
| Garbage in, garbage out | Quality-scored, filterable reference library |

**The CV4Smalls philosophy:** How you work with data matters more than how much of it you have. Five well-curated reference videos produce better coaching than fifty random clips.

---

## Tech Stack

- **[FiftyOne](https://docs.voxel51.com)** — Dataset management, plugin framework, visual App
- **[Twelve Labs Marengo 3.0](https://docs.twelvelabs.io)** — Video embeddings (512-dim) + similarity search
- **[Twelve Labs Pegasus 1.2](https://docs.twelvelabs.io)** — Video-to-text generation for coaching + verification
- **[Twelve Labs Embed API](https://docs.twelvelabs.io)** — Direct embedding extraction for cosine similarity

---

## License

MIT

---

*"Elite athletes have coaches. Everyone else has CoachMe+."*
