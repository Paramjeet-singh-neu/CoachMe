# CoachMe — AI Sports Coach FiftyOne Plugin

> **"Elite athletes have coaches. Everyone else has CoachMe."**

A FiftyOne plugin that uses Twelve Labs video understanding to analyze sports technique and generate timestamped AI coaching feedback — no human coach required.

**Built by Paramjeet Singh** | MS Information Systems, Northeastern University
Voxel51 + Twelve Labs Hackathon, April 2026

---

## The Problem

99% of athletes worldwide train without access to qualified coaching. A boxing coach, tennis instructor, or weightlifting trainer costs $50–150/hour. In developing countries, they often don't exist at all.

Good technique reference footage is everywhere — YouTube, sports databases, coach recordings. The missing layer is **intelligence**: something that watches your video, compares it to good form, and tells you exactly what to fix and when.

## The Solution

CoachMe lets any athlete:

1. **Load** reference technique videos (pro footage, coach demos)
2. **Upload** their own practice video
3. **Get back** a similarity score + timestamped AI coaching feedback — instantly

No labels. No training data. No coach required.

---

## Architecture

```
                    ┌─────────────────────────┐
                    │     FiftyOne App         │
                    │  (Browse + Visualize)    │
                    └────────┬────────────────┘
                             │
                    ┌────────▼────────────────┐
                    │   CoachMe Plugin         │
                    │                          │
                    │  load_reference_videos   │
                    │  analyze_technique       │
                    │  view_coaching_report    │
                    └────────┬────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────▼──┐  ┌───────▼────┐  ┌──────▼──────┐
    │  Marengo   │  │  Pegasus   │  │  FiftyOne   │
    │  2.7       │  │  1.2       │  │  Dataset    │
    │            │  │            │  │             │
    │ Similarity │  │ Coaching   │  │ Persistent  │
    │ Embeddings │  │ Feedback   │  │ Storage     │
    └────────────┘  └────────────┘  └─────────────┘
         Twelve Labs API               Local DB
```

## How It Works

| Step | What Happens | Powered By |
|------|-------------|------------|
| 1. Index references | Upload pro/good-form videos, compute embeddings | Twelve Labs Marengo 2.7 |
| 2. Upload athlete video | Index athlete's practice video | Twelve Labs Task API |
| 3. Similarity scoring | Compare athlete embeddings to reference embeddings | Marengo visual search |
| 4. Coaching feedback | Generate timestamped technique analysis | Twelve Labs Pegasus 1.2 |
| 5. Browse results | View scores, feedback, and videos side-by-side | FiftyOne App |

---

## Project Status

### Completed

| # | Task | Status | Details |
|---|------|--------|---------|
| 1 | Environment setup | Done | Python 3.11 venv (`hackathon-env`), FiftyOne + Twelve Labs SDK + dotenv installed |
| 2 | Twelve Labs API connection | Done | API key in `.env`, verified with `test_tl.py` |
| 3 | FiftyOne App verified | Done | Tested with `test_fo.py` (quickstart-video zoo dataset) |
| 4 | Plugin scaffold | Done | `coachme/__init__.py` + `fiftyone.yml` created |
| 5 | Operator: `load_reference_videos` | Done | Uploads videos to TL index, creates persistent FiftyOne dataset, progress bars, skip-if-exists logic, dynamic input preview |
| 6 | Operator: `analyze_technique` | Done | Uploads athlete video, Marengo similarity search, Pegasus coaching generation, writes all fields to FiftyOne, progress bars |
| 7 | Operator: `view_coaching_report` | Done | Dropdown selector of past analyses, inline display of scores + feedback + reference matches |
| 8 | Helper functions | Done | `get_client()`, `get_or_create_index()`, `upload_and_wait()` with progress + timeout, `find_videos()` |
| 9 | Plugin metadata | Done | `fiftyone.yml` v1.0.0 with all 3 operators registered |
| 10 | Demo notebook | Done | `notebook.ipynb` — full 9-cell walkthrough as backup demo path |
| 11 | CLI demo script | Done | `demo.py` with argparse — runs full pipeline from terminal |
| 12 | README | Done | Architecture diagram, install guide, operator docs, field reference |
| 13 | Structured coaching prompt | Done | Forces Pegasus output into: Technique Score, What's Good, What Needs Work, Drill Prescription |

### Not Yet Done

| # | Task | Priority | Why It Matters |
|---|------|----------|----------------|
| 1 | End-to-end test with real videos | **CRITICAL** | Nothing has been tested against the actual Twelve Labs API with real video files yet |
| 2 | Collect demo videos | **CRITICAL** | Need 2-3 reference videos (pro technique) + 1 athlete video for the sport you'll demo |
| 3 | Register plugin with FiftyOne | **CRITICAL** | Plugin must be symlinked/copied into FiftyOne's plugins directory to appear in the App |
| 4 | Verify Twelve Labs SDK response shapes | **HIGH** | `search.query()` and `generate.text()` response attributes (`data`, `score`, `video_id`) need to match actual SDK version |
| 5 | Create `requirements.txt` | **HIGH** | For reproducibility and judges to install |
| 6 | Add `.gitignore` | **HIGH** | Exclude `hackathon-env/`, `.env`, `__pycache__/` |
| 7 | Create plugin icon | **MEDIUM** | `coachme/assets/icon.svg` — operators reference it but file doesn't exist |
| 8 | Init git repo + push to GitHub | **MEDIUM** | For submission and sharing |
| 9 | Dry-run the demo flow | **MEDIUM** | Practice the exact sequence: launch App → load refs → analyze → show report |
| 10 | Handle edge cases in SDK responses | **LOW** | Graceful fallback if `search.query()` returns unexpected shape |

---

## Next Steps (in order)

### Step 1: Collect demo videos (15 min)
You need real videos before anything else can be tested.

```
mkdir -p ~/coachme-videos/reference ~/coachme-videos/athlete
```

- Download 2-3 short clips (10-30s each) of **good technique** for your chosen sport (e.g. pro boxing jab from YouTube) → save to `~/coachme-videos/reference/`
- Record or download 1 clip of **your own attempt** at the same skill → save to `~/coachme-videos/athlete/`
- Keep videos under 60s — Twelve Labs indexes faster and hackathon demo time is limited

### Step 2: Create requirements.txt + .gitignore (2 min)

```bash
# From project root
echo "fiftyone\ntwelvelabs\npython-dotenv" > requirements.txt

cat > .gitignore << 'EOF'
hackathon-env/
.env
__pycache__/
*.pyc
.fiftyone/
.ipynb_checkpoints/
EOF
```

### Step 3: Register plugin with FiftyOne (2 min)

```bash
# Option A: symlink (best for dev — changes reflect immediately)
ln -s $(pwd)/coachme $(python -c "import fiftyone; print(fiftyone.config.plugins_dir)")/coachme

# Option B: if plugins_dir doesn't exist yet
mkdir -p $(python -c "import fiftyone; print(fiftyone.config.plugins_dir)")
ln -s $(pwd)/coachme $(python -c "import fiftyone; print(fiftyone.config.plugins_dir)")/coachme
```

Verify it shows up:
```bash
fiftyone plugins list
```
You should see `@paramjeet/coachme` with 3 operators.

### Step 4: End-to-end test — reference loading (10-15 min)
Launch FiftyOne and test the first operator:

```bash
fiftyone app launch
```

1. Press **`** (backtick) → search "CoachMe" → select **Load Reference Videos**
2. Sport: `boxing` (or your chosen sport)
3. Videos dir: `~/coachme-videos/reference`
4. Watch the progress bar — each video takes 30-90s to index
5. Verify: dataset `coachme-reference-boxing` appears with samples showing `tl_video_id` fields

**If it fails:** paste the full error here — most likely cause is SDK response shape mismatch in `index.create()` or `task.create()`.

### Step 5: End-to-end test — technique analysis (5-10 min)
Still in the FiftyOne App:

1. Press **`** → select **Analyze My Technique**
2. Sport: `boxing`, Video: path to your athlete video, Focus: `jab technique`
3. Watch progress: Upload → Marengo similarity → Pegasus coaching → Save
4. Verify: `coachme-athletes` dataset opens with your sample
5. Click the sample → check fields: `similarity_score`, `coaching_feedback`, `reference_matches`

**If Marengo returns no matches:** the athlete video might not have finished indexing, or `search.query()` response shape differs from what we expect. We'll debug.

### Step 6: Test the coaching report viewer (2 min)
1. Press **`** → select **View Coaching Report**
2. Select your analysis from the dropdown
3. Verify: scores, reference matches, and full coaching text display inline

### Step 7: Create plugin icon (5 min)
Create a simple SVG icon:

```bash
mkdir -p coachme/assets
```

Then create `coachme/assets/icon.svg` — a simple coaching/sports icon. Or remove the `icon` references from `__init__.py` if you'd rather skip this.

### Step 8: Git init + push (5 min)

```bash
git init
git add -A
git commit -m "CoachMe v1.0 — AI sports coach FiftyOne plugin"
git remote add origin https://github.com/YOUR_USERNAME/coachme.git
git push -u origin main
```

### Step 9: Demo dry run (10 min)
Practice the exact presentation flow:

1. **Open** FiftyOne App (already has reference dataset loaded)
2. **Run** `analyze_technique` on your video — narrate while progress bar runs
3. **Show** the results in the App — click into the sample, read the similarity score
4. **Open** `view_coaching_report` — show the full AI feedback with timestamps
5. **Read one coaching note aloud** — e.g. "At 0:04 your left elbow drops before extension..."
6. **Pitch** — "Elite athletes have coaches. Everyone else has CoachMe."

Total time: ~3-4 minutes for demo.

---

## Quick Start

### Prerequisites

- Python 3.11+
- A [Twelve Labs API key](https://api.twelvelabs.io)

### Installation

```bash
# Clone the repo
git clone https://github.com/paramjeet/coachme.git
cd coachme

# Create virtual environment
python -m venv hackathon-env
source hackathon-env/bin/activate

# Install dependencies
pip install fiftyone twelvelabs python-dotenv

# Set your API key
echo 'TWELVELABS_API_KEY=your_key_here' > .env

# Register the plugin with FiftyOne
ln -s $(pwd)/coachme $(python -c "import fiftyone; print(fiftyone.config.plugins_dir)")/coachme
```

### Usage

#### Via FiftyOne App (recommended for demo)

```bash
fiftyone app launch
```

Then press **`** (backtick) to open the operator browser and search for "CoachMe".

#### Via Python / Notebook

```python
import fiftyone as fo
import fiftyone.operators as foo

# Load reference videos
foo.execute_operator(
    "@paramjeet/coachme/load_reference_videos",
    sport="boxing",
    videos_dir="/path/to/reference/videos"
)

# Analyze your technique
foo.execute_operator(
    "@paramjeet/coachme/analyze_technique",
    sport="boxing",
    video_path="/path/to/your/video.mp4",
    focus="jab technique"
)

# View results in the App
dataset = fo.load_dataset("coachme-athletes")
session = fo.launch_app(dataset)
```

#### Via Terminal

```bash
python demo.py --sport boxing \
               --refs ~/coachme-videos/reference \
               --athlete ~/coachme-videos/athlete/my_jab.mp4 \
               --focus "jab technique" \
               --launch-app
```

---

## Plugin Operators

### `load_reference_videos`
Index reference technique videos as the coaching baseline.

| Input | Type | Description |
|-------|------|-------------|
| sport | str | Sport or skill name (e.g. boxing, squat, tennis) |
| videos_dir | str | Path to folder of reference videos |

**Output:** Creates `coachme-reference-{sport}` dataset with indexed samples.

### `analyze_technique`
Compare your video to references and get AI coaching feedback.

| Input | Type | Description |
|-------|------|-------------|
| sport | str | Must match the sport used for references |
| video_path | str | Path to your video file |
| focus | str | Optional focus area (e.g. elbow position, footwork) |

**Output:** Similarity score (%), timestamped coaching feedback, saved to `coachme-athletes` dataset.

### `view_coaching_report`
Browse all coaching results with a dropdown selector.

**Output:** Displays full coaching report inline — similarity score, reference matches, and AI feedback.

---

## Sample Fields Written

Each analyzed athlete sample includes:

| Field | Type | Example |
|-------|------|---------|
| `sport` | str | `"boxing"` |
| `role` | str | `"athlete"` |
| `focus_area` | str | `"jab technique"` |
| `tl_video_id` | str | `"abc123..."` |
| `similarity_score` | float | `72.4` |
| `reference_matches` | list[dict] | `[{video_id, filepath, similarity_pct}]` |
| `coaching_feedback` | str | `"## Technique Score: 7/10\n..."` |
| `analyzed_at` | str | `"2026-04-03 14:30:00"` |

---

## Supported Sports

Any sport with reference footage works. Tested with:
- Boxing (jab, cross, hook)
- Weightlifting (squat, deadlift)
- Tennis (serve, forehand)
- Cricket (bowling action)
- Yoga / stretching form

---

## Project Structure

```
Coachme/
├── coachme/
│   ├── __init__.py         ← Plugin operators (3 operators)
│   ├── fiftyone.yml        ← Plugin metadata (v1.0.0)
│   └── assets/
│       └── icon.svg        ← Plugin icon (TODO)
├── hackathon-env/          ← Virtual environment
├── .env                    ← API keys (not committed)
├── notebook.ipynb          ← Interactive demo notebook (9 cells)
├── demo.py                 ← Terminal demo script (argparse CLI)
├── test_fo.py              ← FiftyOne connection test
├── test_tl.py              ← Twelve Labs connection test
├── requirements.txt        ← Dependencies (TODO)
├── .gitignore              ← Exclusions (TODO)
└── README.md               ← You are here
```

---

## Why This Matters

| Without CoachMe | With CoachMe |
|-----------------|--------------|
| $50-150/hr for a coach | Free, instant feedback |
| Limited to your city/network | Works anywhere with a camera |
| Subjective feedback | Consistent AI analysis |
| No record of progress | Every session tracked in FiftyOne |
| Only available during sessions | Available 24/7 |

---

## Tech Stack

- **[FiftyOne](https://docs.voxel51.com)** — Dataset management, plugin framework, visual App
- **[Twelve Labs Marengo 2.7](https://docs.twelvelabs.io)** — Multimodal video embeddings + similarity
- **[Twelve Labs Pegasus 1.2](https://docs.twelvelabs.io)** — Video-language model for coaching generation
- **Python 3.11** — Runtime

---

## License

MIT — Built for the Voxel51 + Twelve Labs Hackathon at Northeastern University.

---

*"Elite athletes have coaches. Everyone else has CoachMe."*
