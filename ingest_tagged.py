from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import shutil
import os

load_dotenv()

# Category mapping based on page numbers
def get_category(page):
    if page <= 8:
        return "general"
    elif page <= 18:
        return "permits_and_approvals"
    elif page <= 46:
        return "building_requirements"
    elif page <= 50:
        return "structural_and_safety"
    else:
        return "miscellaneous"

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

# Convert to LangChain Documents with category metadata
documents = []
for page_data in pages:
    if page_data["text"]:
        category = get_category(page_data["page"])
        documents.append(Document(
            page_content=page_data["text"],
            metadata={
                "source": "BBMP Building Bye-Laws 2003",
                "page": page_data["page"],
                "category": category
            }
        ))

print(f"Created {len(documents)} documents")

# Show category distribution
from collections import Counter
cats = Counter([doc.metadata["category"] for doc in documents])
print("\nCategory distribution:")
for cat, count in cats.items():
    print(f"  {cat}: {count} pages")

# Split into chunks
print("\nSplitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# Show chunk category distribution
chunk_cats = Counter([chunk.metadata["category"] for chunk in chunks])
print("\nChunk distribution by category:")
for cat, count in chunk_cats.items():
    print(f"  {cat}: {count} chunks")

# Delete old ChromaDB and create fresh
print("\nStoring in ChromaDB...")
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db", ignore_errors=True)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"local_files_only": True}
)
vectorstore = Chroma.from_documents(
    chunks, embeddings, persist_directory="./chroma_db"
)
print(f"Stored {len(chunks)} chunks in ChromaDB!")
print("\nDone! Category tags added to all chunks.")