# 💬 DocuChat-AI

*Chat with your documents: a RAG application.*

An intelligent document question-answering system powered by RAG (Retrieval-Augmented Generation) that lets you upload multiple documents and chat with them using natural language.

## 📋 Description

This application allows users to upload PDF, DOCX, and TXT files, automatically generates executive summaries, and enables interactive Q&A conversations. Built with `gpt-6-luna` (through OpenRouter or OpenAI), LangChain, and ChromaDB, it provides accurate answers grounded in your document content with real-time streaming responses.

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

- **Python 3.10+**
- **LangChain** - LLM application framework
- **gpt-6-luna** (OpenRouter or OpenAI) - Language model
- **ChromaDB** - Vector database
- **Gradio** - Web UI framework
- **python-dotenv** - Environment management

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher (tested on 3.12)
- An API key: [OpenRouter](https://openrouter.ai/keys) (recommended) or [OpenAI](https://platform.openai.com/api-keys)
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
   The code is written for LangChain 1.x and Gradio 6.

4. **Add your API key**
   ```bash
   cp .env.example .env
   ```
   Then open `.env` and paste your key (see [Use your own API key](#use-your-own-api-key)).

5. **Run the app**
   ```bash
   python app.py
   ```
   The Gradio interface opens in your browser (by default at http://127.0.0.1:7860).

## 🔑 Use your own API key

1. Copy `.env.example` to `.env`.
2. Paste your key after `OPENROUTER_API_KEY=` (recommended) or `OPENAI_API_KEY=`. OpenRouter is used first if both are set.
3. Done. Optionally set `LLM_MODEL=` to use a different chat model (default: `gpt-6-luna`).

Keys are read from your local `.env` only. `.env` is git-ignored, and keys are never stored in the repo.

You can also paste a key into the **API key (optional)** field at the top of the app. It is used for the current session only, kept in memory, and never logged or saved. Keys starting with `sk-or-` go to OpenRouter, any other key goes to OpenAI. **Reset Knowledge Base** clears it.

> Designed for local single-user use. Do not host it for several users without adding per-session keys and documents.

## 💡 Usage

1. Attach one or more PDF, DOCX or TXT files with the 📎 button and press **Send**. Each file gets a short streamed summary.
2. Ask questions about the documents in the text box. Answers come from the retrieved document chunks, and follow-up questions keep the conversation history.
3. Use **Reset Knowledge Base** to clear all documents and start over, and **Clear Chat** to clear only the conversation.

## 📁 Project Structure

```
app.py                  Gradio UI and event handlers
document_processor.py   Document loading, chunking and streamed summaries
rag_pipeline.py         Vector store, memory and the conversational retrieval chain
config.py               API key and model settings, chunking settings
tests/                  Tests (no API key needed): pytest -q
```

## 🧪 Tests

```bash
pip install pytest
pytest -q
```

## 📄 License

MIT, see [LICENSE](LICENSE).