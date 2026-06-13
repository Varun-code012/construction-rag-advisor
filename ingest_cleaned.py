from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load cleaned text file
print("Loading cleaned BBMP text...")
pages = []
current_page = None
current_text = []

with open("bbmp_cleaned.txt", "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("--- PAGE "):
            if current_page and current_text:
                pages.append({
                    "page": current_page,
                    "text": "\n".join(current_text).strip()
                })
            current_page = int(line.replace("--- PAGE ", "").replace(" ---", "").strip())
            current_text = []
        else:
            current_text.append(line.rstrip())

# Add last page
if current_page and current_text:
    pages.append({
        "page": current_page,
        "text": "\n".join(current_text).strip()
    })

print(f"Loaded {len(pages)} pages")

# Convert to LangChain Documents with page metadata
documents = []
for page_data in pages:
    if page_data["text"]:
        documents.append(Document(
            page_content=page_data["text"],
            metadata={
                "source": "BBMP Building Bye-Laws 2003",
                "page": page_data["page"]
            }
        ))

print(f"Created {len(documents)} documents")

# Split into chunks
print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# Store in ChromaDB
print("Creating embeddings and storing in ChromaDB...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"local_files_only": True}
)

# Delete old ChromaDB and create fresh
import shutil, os
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db", ignore_errors=True)

vectorstore = Chroma.from_documents(
    chunks, embeddings, persist_directory="./chroma_db"
)
print(f"Stored {len(chunks)} chunks in ChromaDB!")
print("Done! Run query.py to test.")