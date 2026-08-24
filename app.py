import streamlit as st
from pypdf import PdfReader
from docx import Document

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

    # PDF
    if file_name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

        return text

    # DOCX
    elif file_name.endswith(".docx"):
        document = Document(uploaded_file)

        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

        return text

    # TXT
    elif file_name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")

    return ""


# ==============================
# UI
# ==============================

st.title("⚡ TalentRank AI")
st.caption("AI-Powered Resume Screening Agent")


with st.sidebar:

    st.header("⚙️ Configuration")

    st.info(
        "Upload a Job Description and multiple candidate "
        "resumes to screen candidates automatically."
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
            f"{len(uploaded_files)} resume(s) uploaded successfully."
        )

        for file in uploaded_files:

            st.write(f"📄 {file.name}")


# ==============================
# Screening
# ==============================

st.divider()


if st.button(
    "🚀 Run Resume Screening",
    type="primary",
    use_container_width=True
):

    if not jd_text.strip():

        st.error("Please enter a Job Description.")

    elif not uploaded_files:

        st.error("Please upload at least one resume.")

    else:

        st.success("Inputs validated successfully!")

        st.subheader("🔄 Screening Pipeline")

        # ------------------------------
        # Extract resumes
        # ------------------------------

        extracted_resumes = []

        with st.spinner("Extracting resume text..."):

            for file in uploaded_files:

                text = extract_text_from_file(file)

                extracted_resumes.append({
                    "filename": file.name,
                    "text": text
                })

        st.success(
            f"Successfully extracted "
            f"{len(extracted_resumes)} resume(s)."
        )

        # ------------------------------
        # Show extracted resumes
        # ------------------------------

        st.subheader("📄 Extracted Resume Text")

        for resume in extracted_resumes:

            with st.expander(resume["filename"]):

                if resume["text"].strip():

                    st.text_area(
                        "Extracted text",
                        resume["text"],
                        height=250,
                        key=resume["filename"]
                    )

                else:

                    st.warning(
                        "No text could be extracted from this file."
                    )