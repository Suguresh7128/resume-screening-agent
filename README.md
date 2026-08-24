# TalentRank AI

## Overview
TalentRank AI is a recruiter-facing Resume Screening Agent for the ROOMAN AI Challenge. It parses PDF, DOCX, and TXT resumes, compares them to one job description with TF-IDF, asks Groq for contextual evaluation, and produces an ordered shortlist with reasoning.

## Problem and Solution
Manual screening is slow and inconsistent. This lightweight application gives a reviewer a transparent lexical signal, structured LLM review, adjustable hybrid scoring, candidate details, and CSV/JSON exports. It is intentionally simple enough to explain in an interview.

## Features
- PDF, DOCX paragraph/table, and UTF-8 TXT parsing
- Candidate name, skills, experience, education, qualification, and graduation year extraction
- 10+ resume uploads and one-click demo dataset
- TF-IDF plus cosine similarity
- Structured Groq JSON evaluation with matched/missing skills and rationale
- Configurable NLP/LLM weights, progress feedback, failure-safe NLP-only fallback
- Rankings, metrics, charts, insights, CSV and JSON downloads
- Offline tests for parsing, similarity, ranking, and score calculation

## Architecture
`Job Description + Resumes -> Parser -> Cleaning -> Information Extraction -> TF-IDF -> Groq JSON Evaluation -> Hybrid Score -> Ranking -> Dashboard/Exports`

## Tech Stack
Python 3.11+, Streamlit, scikit-learn, pypdf, python-docx, pandas, and the Groq Python SDK. No React, Node, LangChain, database, or cloud infrastructure is required.

## How It Works
The app reads each file in memory, normalizes text, computes TF-IDF similarity, and sends each complete resume with the JD to Groq. The JSON response is validated before use. A failed API call is marked `LLM Evaluation Failed`; it never receives a fake fit score.

## Scoring Formula
`Final Score = (NLP Weight x NLP Similarity) + (LLM Weight x LLM Fit)`

Default weights are 40% NLP and 60% LLM. The sidebar slider controls NLP weight and automatically sets LLM weight to `100% - NLP`. TF-IDF measures lexical overlap; the LLM adds contextual interpretation. When LLM evaluation fails, the displayed score is explicitly NLP-only.

## Project Structure
- `app.py`: parsers, extraction, similarity, Groq evaluation, ranking, Streamlit UI
- `sample_data/`: sample JD and 12 fictional resumes
- `outputs/`: clearly labeled deterministic example output
- `docs/scoring_method.md`: method details
- `docs/tradeoffs.md`: design tradeoffs and future work
- `tests/`: offline unit tests

## Installation
### Windows
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Linux/macOS
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## API Key Configuration
Set `GROQ_API_KEY` in the environment, or enter it in the password field in the sidebar. The environment variable takes priority. Never put a real key in source, README, Git, logs, or generated files. The sidebar value is used only for the current Streamlit session.

The default model is `llama-3.3-70b-versatile`. Groq model availability can vary by account and over time. The app automatically lists models at run time and switches to an available preferred chat model when the requested model is unavailable. You can also use **Discover available models** in the sidebar to inspect and select a model manually. An unavailable model is reported clearly and never receives a fabricated score.

## Running the App
```bash
python -m streamlit run app.py
```

Click **Load Demo Dataset**, enter a Groq key, then click **Run AI Screening**. The demo contains 12 fictional candidates with varied fit. PDF, DOCX, and TXT uploads can also be used directly.

## Example Output and Exports
`outputs/sample_ranking.csv` and `outputs/sample_ranking.json` are deterministic preview files, clearly labeled as examples and not fabricated Groq output. Live runs provide `candidate_ranking_results.csv` and `candidate_ranking_results.json` download buttons.

## Design Decisions and Tradeoffs
TF-IDF is fast, explainable, deterministic, and offline-testable but lexical. Groq supplies contextual reasoning but adds latency, cost, model availability, and failure risk. The hybrid approach makes both signals visible and keeps API failures honest. See `docs/tradeoffs.md`.

## Limitations
Scanned PDFs need OCR, semantic synonyms are not fully captured, and LLM output depends on the selected model. No hiring decision should be automated from this prototype. Production deployment should add redaction, retention policy, authentication, caching, async rate limiting, skill taxonomy, and bias auditing.

## Responsible AI
This system is an assistive screening tool, not an autonomous hiring decision-maker. Evaluate job-relevant information only. Do not score age, gender, religion, race, nationality, marital status, photographs, or other protected characteristics. A human recruiter remains responsible for the final decision.

## Testing
```bash
python -m pytest -q
```
Tests do not require a Groq API key.

## Challenge Requirements Checklist
- [x] PDF, DOCX, and TXT parsing
- [x] Name, skills, experience, education, qualification, graduation year
- [x] One JD and 10+ resumes
- [x] NLP similarity, LLM evaluation, hybrid scoring, ranking
- [x] Reasoning, matched skills, missing skills, charts, progress
- [x] CSV and JSON export
- [x] Sample JD, 12 sample resumes, sample output
- [x] Scoring method and tradeoff notes
- [x] Fairness guidance and human oversight
- [x] README, reproducible setup, `.gitignore`
- [x] Offline parser and similarity tests
