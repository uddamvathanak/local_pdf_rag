import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
from io import BytesIO
import ollama

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Dict, Any, Optional

# Streamlit page configuration
st.set_page_config(
    page_title="Ollama PDF RAG Streamlit UI",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

@st.cache_resource(show_spinner=True)
def extract_model_names(
    models_info: Dict[str, List[Dict[str, Any]]],
) -> Tuple[str, ...]:
    """
    Extract model names from the provided models information.

    Args:
        models_info (Dict[str, List[Dict[str, Any]]]): Dictionary containing information about available models.

    Returns:
        Tuple[str, ...]: A tuple of model names.
    """
    logger.info("Extracting model names from models_info")
    model_names = tuple(model["name"] for model in models_info["models"])
    logger.info(f"Extracted model names: {model_names}")
    return model_names

def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        List[Dict[str, Any]]: List of document chunks.
    """
    logger.info("Extracting text from PDF")
    try:
        loader = UnstructuredPDFLoader(pdf_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        logger.info("Document split into chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        st.error(f"Error extracting text from PDF: {e}. Ensure poppler is installed and in the system PATH.")
        return []

def create_vector_db(files: List[str]) -> Chroma:
    """
    Create a vector database from a list of PDF files.

    Args:
        files (List[str]): List of paths to the PDF files.

    Returns:
        Chroma: A vector store containing the processed document chunks.
    """
    logger.info("Creating vector DB from files")
    all_chunks = []
    for file in files:
        chunks = extract_text_from_pdf(file)
        if chunks:
            all_chunks.extend(chunks)

    if not all_chunks:
        st.error("No valid PDF content found in the uploaded files.")
        return None

    embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
    vector_db = Chroma.from_documents(
        documents=all_chunks, embedding=embeddings, collection_name="myRAG"
    )
    logger.info("Vector DB created")
    return vector_db

def handle_multiple_file_upload(uploaded_files) -> List[str]:
    """
    Handle the uploaded files, saving them to a temporary directory.

    Args:
        uploaded_files (List[st.UploadedFile]): List of uploaded files.

    Returns:
        List[str]: List of paths to the saved PDF files.
    """
    logger.info(f"Handling multiple file uploads: {[file.name for file in uploaded_files]}")
    temp_dir = tempfile.mkdtemp()
    pdf_paths = []
    try:
        for uploaded_file in uploaded_files:
            pdf_path = os.path.join(temp_dir, uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            pdf_paths.append(pdf_path)
        return pdf_paths
    except Exception as e:
        logger.error(f"Error handling file upload: {e}")
        st.error(f"Error handling file upload: {e}")
        return []

def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
    """
    Process a user question using the vector database and selected language model.

    Args:
        question (str): The user's question.
        vector_db (Chroma): The vector database containing document embeddings.
        selected_model (str): The name of the selected language model.

    Returns:
        str: The generated response to the user's question.
    """
    logger.info(f"Processing question: {question} using model: {selected_model}")
    llm = ChatOllama(model=selected_model, temperature=0)

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate 3
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )

    template = """Answer the question based on the following context:
    {context}
    Question: {question}
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Only provide the answer from the {context}, nothing else.
    Add snippets of the context you used to answer the question.


    In case the answer couldn't be found. advise the user that the answer couldn't be found and here is your general knowledge behind it and give a generic answer unless the user specify the context.
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    logger.info("Question processed and response generated")
    return response

@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    """
    Extract all pages from a PDF file as images.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

    Returns:
        List[Any]: A list of image objects representing each page of the PDF.
    """
    logger.info(f"Extracting all pages as images from file: {file_upload.name}")
    try:
        pdf_pages = []
        with pdfplumber.open(file_upload) as pdf:
            pdf_pages = [page.to_image().original for page in pdf.pages]
        logger.info("PDF pages extracted as images")
        return pdf_pages
    except Exception as e:
        logger.error(f"Error extracting pages as images: {e}")
        st.error(f"Error extracting pages as images: {e}. Ensure poppler is installed and in the system PATH.")
        return []

def delete_vector_db(vector_db: Optional[Chroma]) -> None:
    """
    Delete the vector database and clear related session state.

    Args:
        vector_db (Optional[Chroma]): The vector database to be deleted.
    """
    logger.info("Deleting vector DB")
    try:
        if vector_db is not None:
            vector_db.delete_collection()
            st.session_state.pop("pdf_pages", None)
            st.session_state.pop("file_upload", None)
            st.session_state.pop("vector_db", None)
            st.success("Collection and temporary files deleted successfully.")
            logger.info("Vector DB and related session state cleared")
            st.rerun()
        else:
            st.error("No vector database found to delete.")
            logger.warning("Attempted to delete vector DB, but none was found")
    except Exception as e:
        logger.error(f"Error deleting vector DB: {e}")
        st.error(f"Error deleting vector DB: {e}")

def main() -> None:
    """
    Main function to run the Streamlit application.

    This function sets up the user interface, handles file uploads,
    processes user queries, and displays results.
    """
    st.subheader("🧠 Ollama PDF RAG playground", divider="gray", anchor=False)

    models_info = ollama.list()
    available_models = extract_model_names(models_info)

    col1, col2 = st.columns([1.5, 2])

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None

    if available_models:
        selected_model = col2.selectbox(
            "Pick a model available locally on your system ↓", available_models
        )

    file_uploads = col1.file_uploader(
        "Upload PDF files ↓", type=["pdf"], accept_multiple_files=True
    )

    if file_uploads:
        st.session_state["file_uploads"] = file_uploads
        files = handle_multiple_file_upload(file_uploads)
        if st.session_state["vector_db"] is None and files:
            st.session_state["vector_db"] = create_vector_db(files)
        
        if len(file_uploads) == 1:
            pdf_pages = extract_all_pages_as_images(file_uploads[0])
            st.session_state["pdf_pages"] = pdf_pages

            zoom_level = col1.slider(
                "Zoom Level", min_value=100, max_value=1000, value=700, step=50
            )

            with col1:
                with st.container(height=410, border=True):
                    for page_image in pdf_pages:
                        st.image(page_image, width=zoom_level)

    delete_collection = col1.button("⚠️ Delete collection", type="secondary")

    if delete_collection:
        delete_vector_db(st.session_state["vector_db"])

    with col2:
        message_container = st.container(height=500, border=True)

        for message in st.session_state["messages"]:
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        if prompt := st.chat_input("Enter a prompt here..."):
            try:
                st.session_state["messages"].append({"role": "user", "content": prompt})
                message_container.chat_message("user", avatar="😎").markdown(prompt)

                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":green[processing...]"):
                        if st.session_state["vector_db"] is not None:
                            response = process_question(
                                prompt, st.session_state["vector_db"], selected_model
                            )
                            st.markdown(response)
                        else:
                            st.warning("Please upload a PDF file first.")

                if st.session_state["vector_db"] is not None:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                    )

            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state["vector_db"] is None:
                st.warning("Upload a PDF file to begin chat...")

if __name__ == "__main__":
    main()
