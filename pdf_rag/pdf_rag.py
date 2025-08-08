# 1. Ingest pdf files from ../data
# 2. Extract text from pdf files and split into chunks
# 3. Send chunks to embedding model
# 4. Save embeddings to a vector database
# 5. Perform similarity search on the vector database to find similar documents
# 6. Retrieve the similar docs and present them to the user

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader

doc_path = "../data/hba1c_interpretation.pdf"
model = "gemma3:4b"

# Local PDF file upload
if doc_path:
    loader = UnstructuredPDFLoader(file_path=doc_path)
    data = loader.load()
    print(f"Loaded document from local PDF file: {doc_path}")
else:
    print("Upload a PDF file.")


# Preview first page
content = data[0].page_content
# print(f"Preview of the first page:\n{content[:500]}...")  # is working

# === End of PDF ingestion ===
