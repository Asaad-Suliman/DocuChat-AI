import os
import time
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from config import MODEL, OPENAI_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP


def load_document(file_path):
    """
    Load a document from disk based on its file extension.
    Returns (docs, error_message). If successful, error_message is None.
    """
    ext = file_path.split(".")[-1].lower()

    if ext == "pdf":
        loader = PyPDFLoader(file_path)
    elif ext == "docx":
        loader = UnstructuredWordDocumentLoader(file_path)
    elif ext == "txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        return None, f"Unsupported file type: {ext}"

    try:
        docs = loader.load()
        return docs, None
    except Exception as e:
        return None, f"Error loading document: {str(e)}"


def chunk_documents(docs):
    """
    Split a list of documents into smaller chunks for vector storage.
    Returns a list of chunks.
    """
    splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


def generate_summary_streaming(docs, filename):
    """
    Stream a bullet-point summary of a document using GPT.
    Yields partial strings as the model generates them (for live UI updates).
    """
    try:
        # Limit input to avoid token overflow
        combined_text = "\n".join([doc.page_content for doc in docs[:10]])
        if len(combined_text) > 8000:
            combined_text = combined_text[:8000]

        llm = ChatOpenAI(
            temperature=0.3,
            model_name=MODEL,
            openai_api_key=OPENAI_API_KEY,
            streaming=True
        )

        prompt = f"""You are an expert document analyst. Read the document below and extract the 3-5 most important points.

Rules:
- Output ONLY bullet points. No headers, no intro, no conclusion.
- Each bullet starts with a hyphen (-).
- Each bullet is one clear, direct sentence.

Document filename: {filename}
Document content:
{combined_text}

Generate the bullet points now."""

        # Stream the response back token by token
        summary_text = f"📄 **{filename}**\n"
        yield summary_text

        for chunk in llm.stream(prompt):
            if hasattr(chunk, "content"):
                summary_text += chunk.content
                yield