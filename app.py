import streamlit as st
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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

        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

        return text

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
# UI
# ==============================

st.title("⚡ TalentRank AI")
st.caption(
    "AI-Powered Resume Screening Agent | "
    "TF-IDF + Cosine Similarity"
)


# ==============================
# Sidebar
# ==============================

with st.sidebar:

    st.header("⚙️ Configuration")

    st.info(
        "Upload a Job Description and multiple candidate "
        "resumes to rank candidates."
    )

    st.markdown("---")

    st.subheader("📊 Current Scoring")

    st.write("NLP Similarity: **100%**")


# ==============================
# Input Section
# ==============================

col1, col2 = st.columns(2)


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
# Screening
# ==============================

st.divider()


if st.button(
    "🚀 Run NLP Screening",
    type="primary",
    use_container_width=True
):

    if not jd_text.strip():

        st.error("Please enter a Job Description.")

    elif not uploaded_files:

        st.error("Please upload at least one resume.")

    else:

        # ------------------------------
        # Extract resumes
        # ------------------------------

        resumes = []

        with st.spinner("Extracting resume text..."):

            for file in uploaded_files:

                text = extract_text_from_file(file)

                resumes.append({
                    "filename": file.name,
                    "text": text
                })


        # ------------------------------
        # Validate extraction
        # ------------------------------

        valid_resumes = [
            resume
            for resume in resumes
            if resume["text"].strip()
        ]

        if not valid_resumes:

            st.error(
                "Could not extract text from the uploaded resumes."
            )

            st.stop()


        # ------------------------------
        # TF-IDF scoring
        # ------------------------------

        with st.spinner(
            "Calculating TF-IDF similarity scores..."
        ):

            scores = calculate_similarity(
                jd_text,
                valid_resumes
            )


        # ------------------------------
        # Create ranking
        # ------------------------------

        results = []

        for i, resume in enumerate(valid_resumes):

            results.append({
                "Rank": 0,
                "Candidate": resume["filename"],
                "NLP Similarity": scores[i]
            })


        results.sort(
            key=lambda x: x["NLP Similarity"],
            reverse=True
        )


        for i, result in enumerate(results):

            result["Rank"] = i + 1


        # ------------------------------
        # Display results
        # ------------------------------

        st.success(
            f"Successfully screened {len(results)} candidate(s)."
        )

        st.subheader("🏆 Candidate Ranking")


        # Top candidate
        top_candidate = results[0]

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Candidates Screened",
            len(results)
        )

        metric2.metric(
            "Top Candidate",
            top_candidate["Candidate"]
        )

        metric3.metric(
            "Highest NLP Score",
            f"{top_candidate['NLP Similarity']}%"
        )


        # Ranking table

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )


        # ------------------------------
        # Explanation
        # ------------------------------

        st.subheader("🧠 How the Score Works")

        st.write(
            """
            **TF-IDF (Term Frequency-Inverse Document Frequency)**
            converts the Job Description and each resume into numerical
            vectors based on important words.

            **Cosine Similarity** then measures how closely each resume
            matches the Job Description.

            A higher percentage means greater lexical overlap with the JD.
            """
        )