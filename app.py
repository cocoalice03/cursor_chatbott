from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import anthropic
from typing import List
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import uuid

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure Anthropic
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

# Pinecone settings
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = "n8n"

if not pinecone_api_key or not index_name:
    raise ValueError("Missing Pinecone API key or index name")

print(f"Using index name: {index_name}")

# Init Pinecone SDK v6
print("Using Pinecone SDK v6+")
pc = Pinecone(api_key=pinecone_api_key)

# Use correct dimension based on embedding model
dimension = 1536 if os.getenv("OPENAI_API_KEY") else 384

# Create index if not exists
if index_name not in [i.name for i in pc.list_indexes()]:
    print(f"Creating index {index_name} with dimension {dimension}")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
else:
    print(f"Using existing index: {index_name}")

index = pc.Index(index_name)
print(f"Connected to index: {index_name}")

# OpenAI client or fallback
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
if openai_client:
    print("Using OpenAI embeddings")
else:
    print("OpenAI API key not found, using sentence transformer")

# SentenceTransformer fallback model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text: str):
    if openai_client:
        try:
            response = openai_client.embeddings.create(
                input=[text],
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error using OpenAI embeddings: {e}, falling back")
    return model.encode(text).tolist()

def add_document_to_index(text: str, metadata: dict = None) -> None:
    embedding = get_embedding(text)
    metadata = metadata or {}
    metadata['text'] = text
    doc_id = f"doc_{uuid.uuid4()}"
    index.upsert(vectors=[(doc_id, embedding, metadata)])
    print(f"Added document: {doc_id}")

def get_relevant_context(query: str, top_k: int = 3) -> List[str]:
    embedding = get_embedding(query)
    results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    print("Results from Pinecone:", results)
    contexts = []
    for match in results['matches']:
        meta = match.get("metadata") or {}
        text = meta.get("text") or meta.get("pageContent")
        if text:
            contexts.append(text)
    return contexts

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    try:
        relevant_contexts = get_relevant_context(user_message)
        if not relevant_contexts:
            context_prompt = f"""The user asked: \"{user_message}\"

I don't have any relevant information about this in my database. I can provide a general response, but please note that this information is not from my database:

Please provide a general response to this question."""
        else:
            context_prompt = f"""Here is the relevant information from my database that might help answer the question:

{' '.join(relevant_contexts)}

Based on the above information from my database, please answer the following question: {user_message}

If the information from my database doesn't fully answer the question, please:
1. Provide what the database says
2. Mention you're switching to general knowledge
3. Then add complementary information if needed"""

        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": context_prompt}]
        )
        return jsonify({"response": response.content[0].text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/add_document', methods=['POST'])
def add_document():
    data = request.json
    text = data.get('text', '')
    metadata = data.get('metadata', {})
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    try:
        add_document_to_index(text, metadata)
        return jsonify({'message': 'Document added successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/debug_index')
def debug_index():
    stats = index.describe_index_stats()
    return jsonify(stats)

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')
