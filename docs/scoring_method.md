# Scoring Method

## 1. Resume parsing
`pypdf` extracts PDF pages, `python-docx` extracts DOCX paragraphs and tables, and TXT files are decoded as UTF-8 with replacement for malformed bytes. Empty or corrupted inputs are reported as extraction issues.

## 2. Text preprocessing
Whitespace is normalized while headings and line breaks are retained. The raw resume is kept only in session memory for the current run.

## 3. TF-IDF and cosine similarity
The job description and each resume are represented with TF-IDF unigrams and bigrams. Cosine similarity compares the JD vector with each resume vector and is scaled to 0-100. This is a fast, transparent lexical relevance signal.

## 4. LLM evaluation
Groq receives the complete JD and resume and must return JSON containing candidate facts, matched and missing skills, relevant experience, education, a 0-100 fit score, and rationale. The prompt forbids invention and uses `Not Specified` for missing facts.

## 5. Hybrid scoring
The recruiter chooses the NLP weight. The LLM weight is automatically the remainder:

`Final Score = (NLP Weight x NLP Similarity) + (LLM Weight x LLM Fit)`

The default is 40% TF-IDF and 60% LLM. If an LLM call fails, no artificial score is used; the final score becomes the NLP score and is labeled `NLP-only fallback (LLM failed)`.

## 6. Ranking
Candidates are sorted descending by final score. Thresholds are for shortlist counts only and do not alter scores. Results can be exported as CSV or JSON.
