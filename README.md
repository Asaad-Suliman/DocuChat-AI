# 💬 Chat with Documents - RAG Application

An intelligent document question-answering system powered by RAG (Retrieval-Augmented Generation) that lets you upload multiple documents and chat with them using natural language.

## 📋 Description

This application allows users to upload PDF, DOCX, and TXT files, automatically generates executive summaries, and enables interactive Q&A conversations. Built with OpenAI's GPT-4o-mini, LangChain, and ChromaDB, it provides accurate answers grounded in your document content with real-time streaming responses.

## ✨ Features

- 📄 **Multi-Format Support** - Upload PDF, DOCX, and TXT files
- 🤖 **Auto-Summarization** - Generates bullet-point summaries for each document
- 💬 **Intelligent Chat** - Ask questions and get context-aware answers
- ⚡ **Streaming Responses** - Real-time text generation for natural interaction
- 🧠 **Conversation Memory** - Maintains chat history for follow-up questions
- 📚 **Multi-Document Processing** - Handle multiple documents simultaneously
- 🔍 **Vector Search** - Semantic search using ChromaDB and OpenAI embeddings
- 🎨 **Clean UI** - Intuitive Gradio interface with file management
- 🔄 **Knowledge Base Control** - View processed files and reset system

## 🛠️ Technologies Used

- **Python 3.9+**
- **LangChain** - LLM application framework
- **OpenAI GPT-4o-mini** - Language model
- **ChromaDB** - Vector database
- **Gradio** - Web UI framework
- **python-dotenv** - Environment management

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher (tested on 3.12)
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Asaad-Suliman/DocuChat-AI.git
   cd DocuChat-AI
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```

3. **Install the dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   The versions are pinned to the LangChain 0.3 and Gradio 5 lines the code is written for.

4. **Add your API key**
   ```bash
   cp .env.example .env
   ```
   Then open `.env` and set `GPT_KEY` to your OpenAI API key.

5. **Run the app**
   ```bash
   python app.py
   ```
   The Gradio interface opens in your browser (by default at http://127.0.0.1:7860).

## 💡 Usage

1. Attach one or more PDF, DOCX or TXT files with the 📎 button and press **Send**. Each file gets a short streamed summary.
2. Ask questions about the documents in the text box. Answers come from the retrieved document chunks, and follow-up questions keep the conversation history.
3. Use **Reset Knowledge Base** to clear all documents and start over, and **Clear Chat** to clear only the conversation.

## 📁 Project Structure

```
app.py                  Gradio UI and event handlers
document_processor.py   Document loading, chunking and streamed summaries
rag_pipeline.py         Vector store, memory and the conversational retrieval chain
config.py               Model names and chunking settings
```