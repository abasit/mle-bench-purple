# MLE-Bench Purple Agent

A purple agent that solves [MLE-bench](https://github.com/openai/mle-bench) Kaggle-style competitions.

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- [Git LFS](https://git-lfs.github.com/) (required for mlebench leaderboard data)
- Kaggle API credentials (`~/.kaggle/kaggle.json`)

## Setup

### 1. Clone this repo

```bash
git clone <your-repo-url> mle-bench-purple
cd mle-bench-purple
uv sync
```

### 2. Clone and install mlebench (with Git LFS)

The mlebench package includes leaderboard CSV files tracked by Git LFS. Installing via pip alone will give you LFS pointer files instead of actual data, which breaks grading. You must clone the repo with LFS:

```bash
brew install git-lfs  # macOS, or see https://git-lfs.github.com/
cd ..
git clone https://github.com/openai/mle-bench.git
cd mle-bench
git lfs install
git lfs pull
```

### 3. Install mlebench into the green agent environment

The green agent (evaluator) needs the properly cloned mlebench:

```bash
cd ../mle-bench-green
uv pip install -e ../mle-bench
```

**Important:** If the green agent was previously installed with a pip version of mlebench, the old LFS pointer files may persist in `.venv/`. Fix by copying the real files over:

```bash
cp ../mle-bench/mlebench/competitions/spaceship-titanic/leaderboard.csv \
   .venv/lib/python3.13/site-packages/mlebench/competitions/spaceship-titanic/leaderboard.csv
```

### 4. Set up Kaggle credentials

The green agent needs Kaggle credentials to download competition data. Place your `kaggle.json` at `~/.kaggle`

## Running Locally

### Start the green agent (evaluator)

```bash
cd mle-bench-green
uv run src/server.py --port 9009
```

### Start the purple agent

```bash
cd mle-bench-purple
uv run src/server.py --port 9010
```

### Run an assessment

```bash
python test_assessment.py --green-port 9009 --purple-port 9010 --competition spaceship-titanic
```

This sends an assessment request to the green agent, which downloads the competition data, sends it to the purple agent, and grades the submission.

## How It Works

### Assessment Flow

1. The **green agent** receives an assessment request with a `competition_id` and the purple agent's URL
2. It downloads and prepares the Kaggle competition data, tars the public directory, and sends it to the purple agent along with instructions
3. The **purple agent** receives the tar + instructions, extracts the data, solves the competition, and returns a `submission.csv` as a file artifact
4. The green agent grades the submission against the competition's test set and leaderboard

### Purple Agent Structure

```
src/
├── server.py      # A2A server config and agent card
├── agent.py       # Agent logic (this is where the work happens)
├── executor.py    # A2A request handling
└── messenger.py   # A2A messaging utilities
```

### Validation

The green agent supports submission validation. Before final submission, the purple agent can send a status update with `"validate"` in the text and the CSV attached as a `FilePart`. The green agent will respond with whether the submission format is valid (but not a score).

### Current Status

The skeleton agent submits the `sample_submission.csv` from the competition data as a baseline. The next step is to implement actual ML solving logic (LLM-powered code generation + execution).

## Docker

```bash
docker build -t mle-bench-purple .
docker run -p 9010:9010 mle-bench-purple --host 0.0.0.0 --port 9010
```

## Project Structure

Built as an A2A-compatible competition-solving agent.
