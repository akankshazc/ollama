from pathlib import Path
import logging
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import MultiQueryRetriever
import ollama


# Set up logging
logging.basicConfig(level=logging.INFO)


# Constants
DOC_PATH = "./data/hba1c_interpretation.pdf"
MODEL_NAME = "gemma3:4b"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "pdf_rag"


# Load PDF document
def ingest_pdf(doc_path):
    """Load the PDF document."""
    if not Path(doc_path).exists():
        logging.error(f"Document not found: {doc_path}")
        return None
    else:
        loader = UnstructuredFileLoader(file_path=doc_path)
        data = loader.load()
        logging.info(f"Loaded {len(data)} documents from {doc_path}")
        return data


# Split text into chunks
def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=400)
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Split into {len(chunks)} chunks")
    return chunks


# Create vector store
def create_vector_db(chunks):
    """Create a vector database from the document chunks."""
    # Pull embedding model if not already available

    # ollama.pull(EMBEDDING_MODEL)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        collection_name=VECTOR_STORE_NAME,
    )

    logging.info(f"Created vector store with {len(vector_db)} documents")
    return vector_db


# Create retriever
def create_retriever(vector_db, llm):
    """Create a multi-query retriever."""

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
        vector_db.as_retriever(), llm=llm, prompt=QUERY_PROMPT
    )

    logging.info("Created multi-query retriever")
    return retriever


# Create Chain
def create_chain(retriever, llm):
    """Create the chain"""
    # RAG prompt template
    template = """Answer the question based ONLY on the following context: {context}
    Question: {question}"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = ({"context": retriever, "question": RunnablePassthrough()}
             | prompt
             | llm
             | StrOutputParser())

    logging.info("Created chain for RAG")
    return chain


# main function
def main():
    """Load and process the PDF document."""
    data = ingest_pdf(DOC_PATH)
    if data is None:
        return

    # Split the document into chunks
    chunks = split_documents(data)

    # Create vector database
    vector_db = create_vector_db(chunks)

    # Initialize the LLM
    llm = ChatOllama(model=MODEL_NAME)

    # Create retriever
    retriever = create_retriever(vector_db, llm)

    # Create chain
    chain = create_chain(retriever, llm)

    logging.info("Pipeline created successfully")
    # Define the question
    question = "What is the significance of HBA1C in diabetes management?"

    # Get the response
    res = chain.invoke(input=question)
    print(res)


if __name__ == "__main__":
    main()
