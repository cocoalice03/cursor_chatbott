from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
import os
import google.generativeai as genai
from typing import List
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import uuid
import redis
from flask_session import Session
from datetime import datetime

# Charger les variables d'environnement
load_dotenv()

# Configurer Flask
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key")

# Configurer Flask-Session
app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_KEY_PREFIX'] = 'chatbot_session:'
app.config['SESSION_REDIS'] = redis.from_url('redis://localhost:6379')
Session(app)

# Configurer Gemini (Google API)
genai.configure(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyD5GrbVYpv6r2bAuy1OPbp3fDZnyg8tAvU"))
model = genai.GenerativeModel("gemini-1.5-flash")

# Configuration Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")

if not pinecone_api_key or not index_name:
    raise ValueError("Missing Pinecone API key or index name")

print(f"Using index name: {index_name}")

# Initialiser Pinecone SDK v6
print("Using Pinecone SDK v6+")
pc = Pinecone(api_key=pinecone_api_key)

# Créer l'index si non existant
dimension = 1536 if os.getenv("OPENAI_API_KEY") else 384

if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)
print(f"Connected to index: {index_name}")

# Client OpenAI ou fallback avec SentenceTransformer
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
if openai_client:
    print("Using OpenAI embeddings")
else:
    print("OpenAI API key not found, using sentence transformer")

# Utilisation du modèle SentenceTransformer comme fallback
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

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
    return embedding_model.encode(text).tolist()

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

# Vérifier le quota de l'utilisateur
def check_user_quota():
    today = datetime.today().strftime('%Y-%m-%d')
    if 'date' not in session or session['date'] != today:
        session['date'] = today
        session['question_count'] = 0

    if session.get('question_count', 0) >= 5:
        return False, "🚫 Vous avez atteint la limite de 5 questions pour aujourd'hui."

    session["question_count"] += 1
    return True, ""

# Routes de Flask
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    # Gestion du quota de 50 questions/jour
    today = datetime.today().strftime('%Y-%m-%d')
    if 'date' not in session or session['date'] != today:
        session['date'] = today
        session['question_count'] = 0

    if session.get('question_count', 0) >= 50:
        return jsonify({"response": "🚫 Tu as dépassé la limite de 50 questions pour aujourd'hui. Reviens demain !"})

    session["question_count"] += 1

    data = request.json
    user_message = data.get('message', '')

    # Vérification du quota via la session
    allowed, message = check_user_quota()
    if not allowed:
        return jsonify({"response": f"🚫 {message}"})

    try:
        relevant_contexts = get_relevant_context(user_message)
        if not relevant_contexts:
            context_prompt = f"""Tu es un assistant qui répond uniquement à partir d'une base de données précise. 

L'utilisateur a posé cette question : "{user_message}"

Tu n'as trouvé **aucune information** en rapport avec cette question dans la base de données.
Donc **tu ne dois pas répondre**.

Contente-toi de dire que tu ne peux pas répondre à cette question car elle ne figure pas dans la base de données."""
        else:
            prompt = f"""Voici des informations issues de ma base de données :

{'\n\n'.join(relevant_contexts)}

Réponds à la question suivante uniquement à partir de ces informations : "{user_message}"

⚠️ Tu dois uniquement utiliser les infos présentes dans le contexte. Ne complète pas avec tes connaissances générales."""

        response = model.generate_content(prompt)
        return jsonify({"response": response.text})
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
