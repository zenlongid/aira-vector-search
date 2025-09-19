from fastapi import FastAPI, Request, HTTPException
import redis
import openai
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

#redis connection
redis_host = os.getenv("REDIS_HOST")
if ":" in redis_host:
    redis_host = redis_host.split(":")[0]

r = redis.Redis(
    host=redis_host,
    port=int(os.getenv("REDIS_PORT")),  # Fixed typo: REDIST_PORT -> REDIS_PORT
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=False  # Must be False for binary embeddings
)

#openai connecton
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text):
    # Updated to new OpenAI API
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding, dtype=np.float32).tobytes()

@app.post("/search")
async def search(request: Request):
    data = await request.json()
    query = data.get("question")
    top_k = data.get("top_k", 3)
    
    print(f"Searching for: {query}")
    embedding = get_embedding(query)
    print(f"Embedding generated, size: {len(embedding)}")

    redis_query = f'*=>[KNN {top_k} @embedding $vector AS score]'
    
    try:
        # Use execute_command for vector search
        result = r.execute_command(
            'FT.SEARCH', 
            'kb_index',
            redis_query,
            'PARAMS', '2', 
            'vector', embedding,
            'RETURN', '2', 'text', 'score',
            'DIALECT', '2'
        )
        
        # Parse results
        print(f"Raw result from Redis: {result[:100]}")  # Debug: print first 100 chars
        results = []
        
        # Check if we have results
        if isinstance(result, list) and len(result) > 1:
            # First element is the count
            count = result[0]
            print(f"Found {count} results")
            
            # Parse each result
            for i in range(1, len(result), 2):
                if i < len(result):
                    doc_data = result[i]
                    text_value = None
                    score_value = None
                    
                    for j in range(0, len(doc_data), 2):
                        if doc_data[j] == b'text':
                            text_value = doc_data[j+1].decode('utf-8') if isinstance(doc_data[j+1], bytes) else doc_data[j+1]
                        elif doc_data[j] == b'score':
                            score_value = float(doc_data[j+1])
                    
                    if text_value:
                        results.append({
                            'text': text_value,
                            'score': score_value or 0
                        })
        
        return {"results": results, "debug_count": len(results)}
    
    except Exception as e:
        return {"error": str(e), "results": []}

@app.get("/test")
async def test():
    try:
        # Get a sample document
        keys = r.keys('doc:*')[:3]
        docs = []
        for key in keys:
            doc = r.hgetall(key)
            docs.append({
                "key": key.decode() if isinstance(key, bytes) else key,
                "has_text": b'text' in doc,
                "has_embedding": b'embedding' in doc,
                "text_preview": doc.get(b'text', b'')[:100].decode('utf-8', errors='ignore') if b'text' in doc else None
            })
        return {"sample_docs": docs, "total_keys": len(r.keys('doc:*'))}
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health():
    try:
        # Test Redis connection
        r.ping()
        
        # Check index info
        info = r.execute_command('FT.INFO', 'kb_index')
        num_docs = 0
        for i, val in enumerate(info):
            if val == b'num_docs':
                num_docs = int(info[i+1])
                break
        
        return {
            "status": "healthy",
            "redis": "connected",
            "index": "kb_index",
            "documents": num_docs
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}
