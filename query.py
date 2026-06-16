from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load API key from .env file
load_dotenv()

# Load existing ChromaDB — no re-ingestion
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"local_files_only": True}
)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),  # Loaded from .env file
    temperature=0
)

# Prompt template
def ask(question):
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
    print(f"\nQuestion: {question}")
    print(f"Answer: {response.content}")

# Change this question to test different queries
def ask(question):
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
    
    # Build source citations from metadata
    sources = []
    seen_pages = []
    for doc in docs:
        page = doc.metadata.get("page", "Unknown")
        source = doc.metadata.get("source", "BBMP Building Bye-Laws 2003")
        if page not in seen_pages:
            seen_pages.append(page)
            category = doc.metadata.get("category", "general")
            sources.append(f"  [{len(sources)+1}] {source} — Page {page} [{category}]")

    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {response.content}")
    print(f"\nSources:")
    for source in sources:
        print(source)
    print("-" * 60)
    

#ask("What are the set-back requirements in Table 4 for residential buildings?")
#ask("What documents are required for building plan approval in Bangalore?")
ask("What are the parking requirements for commercial buildings?")
#ask("What are the minimum room dimensions for a residential building?")
#ask("What is the procedure to obtain an occupancy certificate in Bangalore?")
