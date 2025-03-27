#!/bin/bash

PORT=5001

echo "🔍 Recherche de processus sur le port $PORT..."
PID=$(lsof -ti :$PORT)

if [ -n "$PID" ]; then
  echo "🔪 Un processus utilise le port $PORT (PID: $PID), on le termine..."
  kill -9 $PID
else
  echo "✅ Aucun processus ne bloque le port $PORT."
fi

echo "🚀 Redémarrage de l'application Flask..."
source .venv/bin/activate
python app.py
