import streamlit as st
from pathlib import Path
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama

# Set up logging
logging.basicConfig(level=logging.INFO)


# Constants
DOC_PATH = "./data/hba1c_interpretation.pdf"
MODEL_NAME = "gemma3:4b"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "pdf_rag"
PERSIST_DIRECTORY = "./chroma_db"


# Load PDF document
def ingest_pdf(doc_path):
    """Load the PDF document."""
    if not Path(doc_path).exists():
        logging.error(f"Document not found: {doc_path}")
        return None
    else:
        loader = UnstructuredPDFLoader(file_path=doc_path, languages=["en"])
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


@st.cache_resource
def load_vector_db():
    """Load or create the vector database."""
    # Pull embedding model if not already available
    # ollama.pull(EMBEDDING_MODEL)

    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if Path(PERSIST_DIRECTORY).exists():
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY
        )
        logging.info("Loaded existing vector database.")
    else:
        # Load and process the PDF document
        data = ingest_pdf(DOC_PATH)
        if data is None:
            return None

        # Split the documents into chunks
        chunks = split_documents(data)

        # Create a new vector database
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY
        )
        vector_db.persist()
        logging.info("New vector database created and persisted.")

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
    st.title("Document Assistant with RAG")

    # User input for question
    user_input = st.text_input("Ask a question about the document:")

    if user_input:
        with st.spinner("Generating response..."):
            try:
                # Initialize the LLM
                llm = ChatOllama(model=MODEL_NAME)

                # Load or create the vector database
                vector_db = load_vector_db()
                if vector_db is None:
                    st.error("Failed to load or create the vector database.")
                    return

                # Create retriever
                retriever = create_retriever(vector_db, llm)

                # Create the chain
                chain = create_chain(retriever, llm)

                # Get the response
                response = chain.invoke(input=user_input)

                st.markdown("**Assistant:**")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    else:
        st.info("Please enter a question to get started.")


if __name__ == "__main__":
    main()
