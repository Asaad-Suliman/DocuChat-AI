import os
import time
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from config import MODEL, OPENAI_API_KEY


class RAGPipeline:
    """
    Manages the vectorstore, embeddings, memory, and conversation chain.
    All document-based Q&A goes through this class.
    """

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.vectorstore = None
        self.conversation_chain = None
        self.processed_files = []
        self.all_chunks = []

    def add_documents(self, chunks, filename):
        """
        Add new document chunks to the vectorstore.
        Creates the vectorstore on first call, updates it on subsequent calls.
        """
        self.all_chunks.extend(chunks)
        self.processed_files.append(filename)

        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(self.all_chunks, self.embeddings)
        else:
            self.vectorstore.add_documents(chunks)

        # Rebuild the conversation chain with the updated vectorstore
        llm = ChatOpenAI(
            temperature=0.7,
            model_name=MODEL,
            openai_api_key=OPENAI_API_KEY
        )
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=self.vectorstore.as_retriever(),
            memory=self.memory
        )

    def is_ready(self):
        """Returns True if documents have been loaded and the chain is ready."""
        return self.conversation_chain is not None

    def get_relevant_context(self, query, top_k=3):
        """Retrieve the most relevant document chunks for a query."""
        if not self.vectorstore:
            return ""
        retriever = self.vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(query)
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

        llm = ChatOpenAI(
            temperature=0.7,
            model_name=MODEL,
            openai_api_key=OPENAI_API_KEY,
            streaming=True
        )

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
        llm = ChatOpenAI(
            temperature=0.7,
            model_name=MODEL,
            openai_api_key=OPENAI_API_KEY,
            streaming=True
        )

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
        self.memory.clear()

    def get_processed_files(self):
        """Return a formatted string listing all processed files."""
        if not self.processed_files:
            return "No files processed yet."
        return "📁 Processed files:\n" + "\n".join([f"• {f}" for f in self.processed_files])