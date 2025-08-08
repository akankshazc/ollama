# 1. Ingest pdf files from ../data
# 2. Extract text from pdf files and split into chunks
# 3. Send chunks to embedding model
# 4. Save embeddings to a vector database
# 5. Perform similarity search on the vector database to find similar documents
# 6. Retrieve the similar docs and present them to the user

from langchain_community.document_loaders import OnlinePDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import ollama


doc_path = "../data/hba1c_interpretation.pdf"
model = "gemma3:4b"

# Local PDF file upload
if doc_path:
    loader = UnstructuredPDFLoader(file_path=doc_path, languages=["en"])
    data = loader.load()
    print(f"Loaded document from local PDF file: {doc_path}")
else:
    print("Upload a PDF file.")


# Preview first page
content = data[0].page_content
# print(f"Preview of the first page:\n{content[:500]}...")  # is working

# === End of PDF ingestion ===

# Extract test from PDF and split into chunks


# Split and chunk
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
chunks = text_splitter.split_documents(data)

# print(f"Split the document into {len(chunks)} chunks.") # working
# print(f"First chunk:\n{chunks[0]}...")


# === End of PDF chunking ===

# Add chunks to vector database

ollama.pull("nomic-embed-text")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="pdf_rag"
)

print("Added chunks to vector database.")


# === End of vector database creation ===

# Retrieval

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama

from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# set up model to use
llm = ChatOllama(model=model)

# simple technique to generate multiple question from a single question and
# then retrieve documents based on those questions:

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), llm=llm, prompt=QUERY_PROMPT)

# RAG prompt template
template = """Answer the question based ONLY on the following context: {context}
Question: {question}"""

prompt = ChatPromptTemplate.from_template(template)

# RAG chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# === End of retrieval setup ===

# res = chain.invoke(input=("What is the document about?"))
res = chain.invoke(input=("What are the main things to keep in mind while interpreting HBA1c?"))


print(res)