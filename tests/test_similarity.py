import io

from app import calculate_final_score, compute_tfidf_similarity, rank_candidates


def test_tfidf_prefers_matching_resume():
    resumes = [{"text": "Python FastAPI PostgreSQL Docker"}, {"text": "Graphic design typography illustration"}]
    scores = compute_tfidf_similarity("Python FastAPI PostgreSQL", resumes)
    assert scores[0] > scores[1]
    assert 0 <= scores[0] <= 100


def test_empty_similarity_is_safe():
    assert compute_tfidf_similarity("", [{"text": "Python"}]) == [0.0]


def test_hybrid_formula_and_failure_fallback():
    assert calculate_final_score(80, 60, 0.4) == (68.0, "Evaluated")
    assert calculate_final_score(80, None, 0.4) == (80, "NLP-only fallback (LLM failed)")


def test_rank_candidates_descending():
    ranked = rank_candidates([{"Candidate": "B", "Final Score": 30}, {"Candidate": "A", "Final Score": 90}])
    assert [item["Candidate"] for item in ranked] == ["A", "B"]
    assert [item["Rank"] for item in ranked] == [1, 2]
