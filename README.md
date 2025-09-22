# Nextflow Chatbot

A chatbot application for answering questions about Nextflow using RAG (Retrieval-Augmented Generation).

## Local Deployment

### Frontend

The frontend is built with Next.js 15 and React 19, using TypeScript and Tailwind CSS for styling.

**Tech Stack:**
- Next.js 15 with Turbopack
- React 19
- TypeScript
- Tailwind CSS
- AI SDK for streaming chat interface

**Key Files:**
- `frontend/components/Chat.tsx` - Main chat interface component
- `frontend/app/` - Next.js app router pages
- `frontend/package.json` - Dependencies and scripts

**To run locally:**
```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:3000`.

### Backend

The backend is a FastAPI application that provides chat endpoints with RAG capabilities using OpenAI embeddings and completions.

**Tech Stack:**
- FastAPI
- OpenAI API for embeddings and chat completions
- NumPy for vector operations
- LLM Guard for prompt injection protection
- CORS middleware for cross-origin requests

**Key Files:**
- `backend/app/main.py` - Main FastAPI application
- `backend/app/utils.py` - RAG utilities and document processing
- `backend/requirements.txt` - Python dependencies

**To run locally:**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The backend API will be available at `http://localhost:8000`.

These instructions assume that the environment variables have already been populated. For more detailed deployment instructions including Docker setup, see [DOCKER.md](DOCKER.md).

## Dummy Deployment

The backend of this application has been deployed using [Railway](https://railway.com/).
The frontend of this application has been deployed using [Vercel](https://vercel.com)

To access a live demo of this application, please click this link [here](https://nextflow-chatbot.vercel.app/)