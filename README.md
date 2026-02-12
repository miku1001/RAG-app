# RAG Web Application

A simple Retrieval Augmented Generation (RAG) web application that allows users to upload PDF documents and ask questions based on the content. The system uses semantic search to retrieve relevant context and generates accurate answers grounded only in the provided documents.

## Tech Stack

### Core Technologies
- **Python** - Programming language
- **Streamlit** - Web application framework
- **LangChain** - RAG pipeline orchestration

### Libraries & Models
- **langchain-groq** - Groq LLM integration (llama-3.3-70b-versatile)
- **langchain-community** - Document loaders (PyPDFLoader)
- **langchain-huggingface** - Embeddings (sentence-transformers/all-MiniLM-L6-v2)
- **langchain-chroma** - Vector database for document storage
- **python-dotenv** - Environment variable management


## How to Use

1. **Upload PDF Document**
   - Click "Browse files" and select a PDF file
   - Click "Initialize RAG System" to process the document
   - Wait for the success message

2. **Ask Questions**
   - Enter your question in the text area on the left
   - Click "Get Answer" button
   - View the AI-generated answer on the right side

3. **Upload New Document**
   - Simply upload a new PDF file
   - Click "Initialize RAG System" again
   - Previous data will be automatically cleared

## Features

- ✅ PDF document upload and processing
- ✅ Semantic search with embeddings
- ✅ Context-aware question answering
- ✅ Strict grounding (no hallucinations)
- ✅ Automatic cleanup of old data

## Notes

- The system only answers based on the uploaded document content
- If the answer is not in the document, it will respond with "I don't know"
- Documents are chunked with 2000 character chunks and 100 character overlap
