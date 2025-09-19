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
    
    embedding = get_embedding(query)

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
        results = []
        
        # Check if we have results
        if isinstance(result, list) and len(result) > 0:
            # First element is the count
            count = result[0]
            
            # Parse each result (doc_id at odd indices, fields at even indices after first)
            i = 1
            while i < len(result):
                if i+1 < len(result):
                    doc_id = result[i]
                    doc_fields = result[i+1]
                    
                    text_value = None
                    score_value = None
                    
                    # Parse the fields array
                    for j in range(0, len(doc_fields), 2):
                        if j+1 < len(doc_fields):
                            field_name = doc_fields[j]
                            field_value = doc_fields[j+1]
                            
                            if field_name == b'text':
                                text_value = field_value.decode('utf-8', errors='ignore') if isinstance(field_value, bytes) else field_value
                            elif field_name == b'score':
                                score_value = float(field_value) if isinstance(field_value, bytes) else field_value
                    
                    if text_value:
                        results.append({
                            'text': text_value,
                            'score': score_value or 0
                        })
                
                i += 2  # Move to next doc_id/fields pair
        
        return {"results": results}
    
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
