from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
load_dotenv()

# Step A — Load the PDF
print("Loading PDF...")
loader = PyPDFLoader("bbmp.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# Step B — Split into chunks
print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# Step C — Create embeddings and store in ChromaDB
print("Creating embeddings and storing in ChromaDB...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"local_files_only": True}
)
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
print("Stored in ChromaDB!")

# Step D — Set up Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0  # Add this line
)

# Step E — Set up retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# Step F — Ask a question
print("\nAsking question...")
question = "What are the set-back requirements in Table 4 for residential buildings?"

# Retrieve relevant chunks
docs = retriever.invoke(question)
context = "\n\n".join([doc.page_content for doc in docs])

# Send to Groq with context
# Send to Groq with improved prompt template
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