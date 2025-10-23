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

- Python 3.9 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- pip package manager

### Installation

1. **Clone or download the repository**

2. **Create a virtual environment** (recommended):
```bash