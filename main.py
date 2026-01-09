from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import streamlit as st

# ---------- EXTRACT KEY ----------
key =st.secrets["OPENAI_KEY"]

# ---------- SET ENV + OPENAI ----------


# ---- CONFIG ----
PERSIST_DIR = "prakrit_embedding"
COLLECTION_NAME = "prakrit_dict"

# ---- LOAD EMBEDDINGS ----
embedding_function = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key= st.secrets["OPENAI_KEY"]
)

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_function
)

# ---- GPT-5.2 MODEL ----
gpt_5_2 = ChatOpenAI(
    model="gpt-5.2",
    temperature=0,
    openai_api_key=st.secrets["OPENAI_KEY"]
)

import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI



# ---------------- RETRIEVAL ----------------
def retrieve_prakrit_context(prakrit_word, k=5):
    results = vectorstore.similarity_search(prakrit_word, k=k)
    return "\n\n".join([r.page_content for r in results])

# ---------------- TRANSLATION ----------------
def prakrit_to_english(prakrit_word: str) -> str:
    context = retrieve_prakrit_context(prakrit_word)

    prompt = f"""
You are a Prakrit-to-English-to-Hindi Dictionary Translator.

Rules:
- Use ONLY the dictionary context provided
- Translate the Prakrit word into English and Hindi
- If multiple meanings exist, list them clearly
- If the word is not found, say:
  "The meaning of this Prakrit word is not available in the dictionary."
- Do NOT guess

Dictionary Context:
-------------------
{context}
-------------------

Prakrit Word:
{prakrit_word}
"""

    response = gpt_5_2.invoke(prompt)
    return response.content

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Prakrit Dictionary", layout="centered")

st.title("📖 Prakrit Dictionary Translator")
st.write("Translate **Prakrit → English → Hindi** using a vector-based dictionary.")

prakrit_word = st.text_input("Enter Prakrit Word", placeholder="उदाहरण: अक्खममाण")

if st.button("Translate"):
    if not prakrit_word.strip():
        st.warning("Please enter a Prakrit word.")
    else:
        with st.spinner("Searching dictionary..."):
            result = prakrit_to_english(prakrit_word)
            st.success("Translation Found")
            st.markdown(result)

