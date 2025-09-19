from fastapi import FastAPI, Request
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
        
        return {"results": results}
    
    except Exception as e:
        return {"error": str(e), "results": []}
