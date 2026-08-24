import streamlit as st
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import json


# ==============================
# Page Configuration
# ==============================

st.set_page_config(
    page_title="TalentRank AI",
    page_icon="⚡",
    layout="wide"
)


# ==============================
# Resume Text Extraction
# ==============================

def extract_text_from_file(uploaded_file):

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):

        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

        return text

    elif file_name.endswith(".docx"):

        document = Document(uploaded_file)

        return "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

    elif file_name.endswith(".txt"):

        return uploaded_file.read().decode("utf-8")

    return ""


# ==============================
# TF-IDF Similarity
# ==============================

def calculate_similarity(jd_text, resumes):

    documents = [jd_text] + [
        resume["text"] for resume in resumes
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=2000
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity_scores = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:]
    ).flatten()

    return [
        round(float(score) * 100, 2)
        for score in similarity_scores
    ]


# ==============================
# Groq LLM Evaluation
# ==============================

def evaluate_with_llm(client, jd_text, resume_text):

    system_prompt = """
You are an expert technical recruiter.

Evaluate a candidate resume against a job description.

Return ONLY valid JSON using exactly this structure:

{
    "matched_skills": [],
    "missing_skills": [],
    "experience_years": "",
    "education": "",
    "fit_score": 0,
    "rationale": ""
}

Rules:

1. matched_skills:
   List important skills present in both the JD and resume.

2. missing_skills:
   List important JD skills that are missing or not clearly demonstrated.

3. experience_years:
   Extract the candidate's relevant professional experience.
   If unclear, return "Not Specified".

4. education:
   Extract the candidate's highest relevant education.

5. fit_score:
   Integer from 0 to 100 representing overall suitability.

6. rationale:
   Give a concise 2-3 sentence explanation.

Do not invent information.
"""

    user_prompt = f"""
JOB DESCRIPTION:

{jd_text}

CANDIDATE RESUME:

{resume_text}
"""

    try:

        response = client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],

            temperature=0.1,

            response_format={
                "type": "json_object"
            }
        )

        content = response.choices[0].message.content

        return json.loads(content)

    except Exception as e:

        return {
            "matched_skills": [],
            "missing_skills": [],
            "experience_years": "Not Specified",
            "education": "Not Specified",
            "fit_score": 0,
            "rationale": f"LLM evaluation failed: {str(e)[:150]}"
        }


# ==============================
# Sidebar
# ==============================

with st.sidebar:

    st.header("⚙️ Configuration")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_..."
    )

    st.caption(
        "Your API key is used only for this session "
        "and is not stored by the application."
    )

    st.markdown("---")

    st.subheader("⚖️ Scoring Weights")

    nlp_weight = st.slider(
        "NLP Similarity",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.05
    )

    llm_weight = 1.0 - nlp_weight

    st.write(
        f"LLM Evaluation: **{llm_weight:.0%}**"
    )


# ==============================
# Main UI
# ==============================

st.title("⚡ TalentRank AI")

st.caption(
    "AI-Powered Resume Screening Agent | "
    "TF-IDF + Groq LLM"
)


col1, col2 = st.columns(2)


# ==============================
# Job Description
# ==============================

with col1:

    st.subheader("📋 Job Description")

    jd_text = st.text_area(
        "Paste the Job Description",
        height=300,
        placeholder=(
            "Example:\n\n"
            "Backend Developer\n\n"
            "Requirements:\n"
            "- Python\n"
            "- FastAPI\n"
            "- PostgreSQL\n"
            "- REST APIs\n"
            "- Docker"
        )
    )


# ==============================
# Resume Upload
# ==============================

with col2:

    st.subheader("📄 Candidate Resumes")

    uploaded_files = st.file_uploader(
        "Upload resumes",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.success(
            f"{len(uploaded_files)} resume(s) uploaded."
        )

        for file in uploaded_files:

            st.write(f"📄 {file.name}")


# ==============================
# Run Screening
# ==============================

st.divider()


if st.button(
    "🚀 Run AI Screening",
    type="primary",
    use_container_width=True
):

    if not api_key:

        st.error(
            "Please enter your Groq API key in the sidebar."
        )

        st.stop()

    if not jd_text.strip():

        st.error(
            "Please enter a Job Description."
        )

        st.stop()

    if not uploaded_files:

        st.error(
            "Please upload at least one resume."
        )

        st.stop()


    # ==============================
    # Initialize Groq
    # ==============================

    client = Groq(api_key=api_key)


    # ==============================
    # Extract Resumes
    # ==============================

    resumes = []

    with st.spinner("Extracting resume text..."):

        for file in uploaded_files:

            text = extract_text_from_file(file)

            if text.strip():

                resumes.append({
                    "filename": file.name,
                    "text": text
                })


    if not resumes:

        st.error(
            "Could not extract text from the uploaded resumes."
        )

        st.stop()


    # ==============================
    # TF-IDF
    # ==============================

    with st.spinner(
        "Calculating NLP similarity..."
    ):

        nlp_scores = calculate_similarity(
            jd_text,
            resumes
        )


    # ==============================
    # LLM Evaluation
    # ==============================

    results = []

    progress = st.progress(0)

    for i, resume in enumerate(resumes):

        with st.spinner(
            f"AI evaluating {resume['filename']}..."
        ):

            evaluation = evaluate_with_llm(
                client,
                jd_text,
                resume["text"]
            )


        final_score = round(
            (nlp_scores[i] * nlp_weight)
            +
            (
                evaluation["fit_score"]
                * llm_weight
            ),
            2
        )


        results.append({

            "Candidate": resume["filename"],

            "Final Score": final_score,

            "NLP Similarity": nlp_scores[i],

            "LLM Fit Score": evaluation["fit_score"],

            "Experience": evaluation.get(
                "experience_years",
                "Not Specified"
            ),

            "Education": evaluation.get(
                "education",
                "Not Specified"
            ),

            "Matched Skills": ", ".join(
                evaluation.get(
                    "matched_skills",
                    []
                )
            ),

            "Missing Skills": ", ".join(
                evaluation.get(
                    "missing_skills",
                    []
                )
            ),

            "Rationale": evaluation.get(
                "rationale",
                ""
            )
        })


        progress.progress(
            (i + 1) / len(resumes)
        )


    # ==============================
    # Ranking
    # ==============================

    results.sort(
        key=lambda x: x["Final Score"],
        reverse=True
    )


    for i, result in enumerate(results):

        result["Rank"] = i + 1


    # ==============================
    # Results
    # ==============================

    st.divider()

    st.subheader(
        "🏆 AI Candidate Ranking"
    )


    top = results[0]


    metric1, metric2, metric3 = st.columns(3)


    metric1.metric(
        "Candidates Screened",
        len(results)
    )


    metric2.metric(
        "Top Candidate",
        top["Candidate"]
    )


    metric3.metric(
        "Highest Score",
        f"{top['Final Score']}%"
    )


    # ==============================
    # Ranking Table
    # ==============================

    st.dataframe(

        [
            {
                "Rank": r["Rank"],
                "Candidate": r["Candidate"],
                "Final Score": r["Final Score"],
                "NLP Score": r["NLP Similarity"],
                "LLM Score": r["LLM Fit Score"],
                "Experience": r["Experience"],
                "Education": r["Education"]
            }

            for r in results
        ],

        use_container_width=True,

        hide_index=True
    )


    # ==============================
    # Candidate Details
    # ==============================

    st.subheader(
        "🔍 Candidate Evaluations"
    )


    for result in results:

        with st.expander(
            f"#{result['Rank']} | "
            f"{result['Candidate']} — "
            f"{result['Final Score']}%"
        ):

            st.write(
                f"**Experience:** "
                f"{result['Experience']}"
            )

            st.write(
                f"**Education:** "
                f"{result['Education']}"
            )

            st.write(
                f"**Matched Skills:** "
                f"{result['Matched Skills'] or 'None identified'}"
            )

            st.write(
                f"**Missing Skills:** "
                f"{result['Missing Skills'] or 'None identified'}"
            )

            st.write(
                f"**AI Reasoning:** "
                f"{result['Rationale']}"
            )