import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# Page config
st.set_page_config(
    page_title="BBMP Building Code Advisor",
    page_icon="🏗️",
    layout="centered"
)

# Title
st.title("🏗️ Construction & Building Code Advisor")
st.markdown("Ask any question about **BBMP Building Bye-Laws 2003** and get instant answers with source citations.")
st.divider()

# Load pipeline — only once using cache
@st.cache_resource
def load_pipeline():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"local_files_only": True}
    )
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )
    return retriever, llm

retriever, llm = load_pipeline()

# Prompt template
def get_answer(question, retriever, llm):
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""You are an expert assistant for BBMP Building Bye-Laws and Karnataka construction regulations.

Your job is to answer questions strictly based on the BBMP context provided below.

STRICT RULES:
1. Answer ONLY from the context provided. Do not use outside knowledge.
2. Always mention the specific section number or table number your answer comes from.
3. If the answer is not found in the context, say exactly: "This information is not available in the provided BBMP bye-laws section. Please refer to the full BBMP Building Bye-Laws 2003 document."
4. Give structured answers using bullet points where applicable.
5. Use simple, plain English so property owners and contractors can understand.
6. Never guess or make up regulations.

CONTEXT FROM BBMP BUILDING BYE-LAWS 2003:
{context}

QUESTION: {question}

ANSWER:"""

    response = llm.invoke(prompt)

    # Build citations
    sources = []
    seen_pages = []
    for doc in docs:
        page = doc.metadata.get("page", "Unknown")
        source = doc.metadata.get("source", "BBMP Building Bye-Laws 2003")
        category = doc.metadata.get("category", "general")
        if page not in seen_pages:
            seen_pages.append(page)
            sources.append({
                "page": page,
                "source": source,
                "category": category
            })

    return response.content, sources

# Demo queries
st.markdown("**Try a demo query:**")
demo_queries = [
    "What are the set-back requirements in Table 4 for residential buildings?",
    "What documents are required for building plan approval in Bangalore?",
    "What are the parking requirements for commercial buildings?",
    "What are the minimum room dimensions for a residential building?",
    "What is the procedure to obtain an occupancy certificate in Bangalore?"
]
selected_demo = st.selectbox("Select a demo query", ["-- Select --"] + demo_queries)

# Text input
st.markdown("**Or type your own question:**")
user_question = st.text_input("Enter your question here", placeholder="e.g. What are the setback rules for a residential plot?")

# Determine which question to use
question = None
if selected_demo != "-- Select --":
    question = selected_demo
if user_question:
    question = user_question

# Answer button
# Answer button
if st.button("Get Answer 🔍", key="answer_btn"):
    if question:
        with st.spinner("Searching BBMP Building Bye-Laws..."):
            answer, sources = get_answer(question, retriever, llm)

        st.markdown("### 📋 Question")
        st.info(question)

        st.markdown("### ✅ Answer")
        st.markdown(answer)

        st.markdown("### 📖 Sources")
        for i, src in enumerate(sources):
            st.markdown(f"**[{i+1}]** {src['source']} — Page {src['page']} `{src['category']}`")

        st.divider()
        st.caption("⚠️ This tool is for guidance only. Always consult a qualified professional for legal compliance.")
    else:
        st.warning("Please select a demo query or type your own question.")

