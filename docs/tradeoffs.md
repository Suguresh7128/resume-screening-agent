# Tradeoffs and Limitations

## Why TF-IDF?
TF-IDF is fast, explainable, simple to test offline, and requires no embeddings infrastructure. Its limitations are lexical matching, sensitivity to wording, and weak synonym/semantic understanding.

## Why an LLM?
An LLM can interpret related skills, internship relevance, experience context, education, and gaps, while producing recruiter-readable reasoning. It introduces API dependency, latency, cost and rate limits, model availability changes, malformed responses, and possible evaluation inconsistency.

## Why hybrid scoring?
TF-IDF provides a reproducible anchor and LLM review provides context. A visible weight slider lets a recruiter choose how much to trust each signal. API failures remain visible and are never converted into a fabricated score.

## Responsible use
This is an assistive screening tool, not an autonomous hiring decision-maker. Use job-relevant information only and audit outcomes for disparate impact. Do not score protected characteristics.

## Future improvements
- Sentence embeddings and a vector index for semantic retrieval
- Structured skill taxonomy and normalized seniority extraction
- Second-pass reranking and recruiter feedback loops
- OCR for scanned/image-only resumes
- Bias auditing, redaction, and privacy retention controls
- Caching and asynchronous, bounded LLM calls
