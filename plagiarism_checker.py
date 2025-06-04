import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Must be first Streamlit command in your script
st.set_page_config(page_title="Plagiarism Checker", layout="centered")

# Load the model once (this can take a moment)
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# UI
st.title("🧠 Plagiarism Checker using NLP")
st.write("Compare two pieces of text to find semantic similarity.")

# Text input fields
text1 = st.text_area("Enter First Text", height=200)
text2 = st.text_area("Enter Second Text", height=200)

if st.button("🔍 Check Similarity"):
    if text1.strip() == "" or text2.strip() == "":
        st.warning("Please enter both texts.")
    else:
        # Embed both texts
        embeddings = model.encode([text1, text2], convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        percentage = similarity_score.item() * 100

        # Display result
        st.metric(label="Similarity Score", value=f"{percentage:.2f}%")

        # Interpretation
        if percentage > 85:
            st.error("⚠️ High similarity detected — possible plagiarism.")
        elif percentage > 60:
            st.warning("⚠️ Moderate similarity — review recommended.")
        else:
            st.success("✅ Low similarity — likely original.")
