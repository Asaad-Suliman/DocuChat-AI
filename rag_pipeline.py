import os
import time
from langchain_chroma import Chroma
# LangChain 1.0 moved the legacy chains and memory classes into langchain-classic.
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from config import get_chat_model, get_embeddings, resolve_llm_settings


class RAGPipeline:
    """
    Manages the vectorstore, embeddings, memory, and conversation chain.
    All document-based Q&A goes through this class.
    """

    def __init__(self):
        # Key pasted in the UI. Memory only: never logged, never written to disk.
        self.api_key = ""
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.vectorstore = None
        self.conversation_chain = None
        self.processed_files = []
        self.all_chunks = []

    def settings(self):
        """Provider settings for the current key. Raises MissingKeyError if there is none."""
        return resolve_llm_settings(self.api_key)

    def set_api_key(self, key):
        """Use a key pasted in the UI. Re-embeds loaded chunks if the key changed."""
        key = (key or "").strip()
        if key == self.api_key:
            return
        self.api_key = key
        if self.all_chunks:
            self.vectorstore = Chroma.from_documents(self.all_chunks, get_embeddings(self.settings()))
            self._build_chain()

    def _build_chain(self):
        llm = get_chat_model(self.settings(), temperature=0.7)
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=self.vectorstore.as_retriever(),
            memory=self.memory
        )

    def add_documents(self, chunks, filename):
        """
        Add new document chunks to the vectorstore.
        Creates the vectorstore on first call, updates it on subsequent calls.
        """
        # Embed first: if it fails, the file must not count as processed.
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(self.all_chunks + chunks, get_embeddings(self.settings()))
        else:
            self.vectorstore.add_documents(chunks)
        self.all_chunks.extend(chunks)
        self.processed_files.append(filename)

        # Rebuild the conversation chain with the updated vectorstore
        self._build_chain()

    def is_ready(self):
        """Returns True if documents have been loaded and the chain is ready."""
        return self.conversation_chain is not None

    def get_relevant_context(self, query, top_k=3):
        """Retrieve the most relevant document chunks for a query."""
        if not self.vectorstore:
            return ""
        retriever = self.vectorstore.as_retriever()
        docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs[:top_k]])

    def stream_response(self, message, chat_history):
        """
        Stream a response to the user's message using RAG context.
        Yields (full_response_so_far, updated_chat_history) tuples.
        """
        context = self.get_relevant_context(message)

        history_text = ""
        if len(chat_history) > 1:
            for human, assistant in chat_history[-6:-1]:
                if human and assistant:
                    history_text += f"Human: {human}\nAssistant: {assistant}\n\n"

        prompt = f"""Answer the user's question based on the document context below.

Previous conversation:
{history_text}

Relevant document content:
{context}

User question: {message}

Provide a clear, accurate answer based on the document context."""

        llm = get_chat_model(self.settings(), temperature=0.7, streaming=True)

        full_response = ""
        for chunk in llm.stream(prompt):
            if hasattr(chunk, "content"):
                full_response += chunk.content
                chat_history[-1] = (message, full_response)
                yield full_response, chat_history
                time.sleep(0.02)

    def stream_direct_response(self, message, chat_history):
        """
        Stream a direct LLM response when no documents are loaded.
        """
        llm = get_chat_model(self.settings(), temperature=0.7, streaming=True)

        full_response = ""
        for chunk in llm.stream(message):
            if hasattr(chunk, "content"):
                full_response += chunk.content
                chat_history[-1] = (message, full_response)
                yield full_response, chat_history
                time.sleep(0.02)

    def reset(self):
        """Clear all documents and reset the pipeline to its initial state."""
        self.vectorstore = None
        self.conversation_chain = None
        self.processed_files = []
        self.all_chunks = []
        self.api_key = ""
        self.memory.clear()

    def get_processed_files(self):
        """Return a formatted string listing all processed files."""
        if not self.processed_files:
            return "No files processed yet."
        return "📁 Processed files:\n" + "\n".join([f"• {f}" for f in self.processed_files])