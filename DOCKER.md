# Docker Setup for Nextflow Chatbot

This guide explains how to run the Nextflow Chatbot application using Docker.

## Prerequisites

- Docker and Docker Compose installed on your system
- OpenAI API key (optional - app will run in mock mode without it)

## Quick Start

1. **Clone and navigate to the project directory**
   ```bash
   cd /path/to/folder
   ```

2. **Set up environment variables**
   ```bash
   # For Docker Compose (recommended)
   cp .env.example .env

   # OR for local backend development
   cp backend/.env.example backend/.env

   # Edit the .env file with your OpenAI API key and preferences
   ```

3. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

4. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Status: http://localhost:8000/api/status

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | Your OpenAI API key (optional) |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `GEN_MODEL` | `gpt-4o-mini` | OpenAI generation model |
| `TOP_K` | `3` | Maximum documents to retrieve |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum similarity score for RAG |
| `NEXT_PUBLIC_BACKEND_URL` | `http://localhost:8000` | Backend URL for frontend |

## Docker Commands

### Development Mode
```bash
# Build and start services
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production Mode
```bash
# Build optimized images
docker-compose -f docker-compose.yml up --build

# Run with restart policies
docker-compose up -d
```

### Individual Services

**Backend only:**
```bash
cd backend
docker build -t nextflow-chatbot-backend .
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key nextflow-chatbot-backend
```

**Frontend only:**
```bash
cd frontend
docker build -t nextflow-chatbot-frontend .
docker run -p 3000:3000 -e NEXT_PUBLIC_BACKEND_URL=http://localhost:8000 nextflow-chatbot-frontend
```

## Troubleshooting

### Common Issues

1. **Port conflicts**: Change ports in docker-compose.yml if 3000 or 8000 are in use
2. **Build failures**: Ensure Docker has enough memory allocated
3. **CORS issues**: Backend allows localhost:3000, 127.0.0.1:3000, and 0.0.0.0:3000
4. **Startup timing**: Frontend waits for backend health check before starting

### Health Checks

The application includes health checks for both services with dependency management:
- Backend health check runs every 10s with 30s startup grace period
- Frontend waits for backend to be healthy before starting
- Both services have restart policies

```bash
# Check service status
docker-compose ps

# View health check logs
docker-compose logs backend
docker-compose logs frontend
```

### Logs and Debugging

```bash
# Follow all logs
docker-compose logs -f

# Follow specific service logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Execute commands in running containers
docker-compose exec backend bash
docker-compose exec frontend sh
```

## Mock Mode

If no OpenAI API key is provided, the application will automatically run in mock mode:
- Responses echo the user's input: "Your message was: [user input]"
- No external API calls are made
- All features work except actual AI responses
- Frontend shows "⚠️ Mock Mode" notification

## Security Features

The application includes basic security measures:
- **Prompt Injection Protection**: Uses LLM Guard to detect malicious prompts
- **CORS Configuration**: Restricts cross-origin requests to allowed origins
- **Health Monitoring**: Automated startup testing with smoke tests

## Stopping the Application

```bash
# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Stop and remove images
docker-compose down --rmi all
```