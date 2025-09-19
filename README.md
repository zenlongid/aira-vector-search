# AIRA Vector Search API

FastAPI service for semantic search over PropSuite FAQ documents using Redis Vector Search and OpenAI embeddings.

## Endpoints

- `POST /search` - Search for relevant FAQs
  - Body: `{"question": "your question", "top_k": 3}`
  - Returns: Top k most relevant FAQ documents

## Deployment

Configured for Railway deployment with environment variables:
- REDIS_HOST
- REDIS_PORT
- REDIS_PASSWORD
- OPENAI_API_KEY