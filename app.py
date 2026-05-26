import streamlit as st
from transformers import pipeline

# Load NER model
@st.cache_resource
def load_model():
    return pipeline(
        "token-classification",
        model="Jean-Baptiste/roberta-large-ner-english",
        aggregation_strategy="simple"
    )

ner = load_model()

st.title("Named Entity Recognition (NER)")
st.write("Extract people, organizations, locations, and more.")

text = st.text_area(
    "Enter Text",
    "Elon Musk is the CEO of Tesla and lives in Texas."
)

if st.button("Analyze"):
    results = ner(text)

    if results:
        st.subheader("Detected Entities")

        for entity in results:
            st.write(
                f"{entity['word']} → "
                f"{entity['entity_group']} "
                f"(Score: {entity['score']:.2f})"
            )
    else:
        st.warning("No entities found.")