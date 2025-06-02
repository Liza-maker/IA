from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import fitz  # PyMuPDF
import requests
from flask_cors import CORS


app = Flask(__name__)
CORS(app)  
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text, max_words=500):
    words = text.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def generate_summary(prompt):
    response = requests.post(OLLAMA_URL, json={
        "model": "llama3.1",
        "prompt": prompt,
        "stream": False
    })
    result = response.json()
    return result.get("response", "").strip()

@app.route('/upload', methods=['POST'])
def upload():
    print("Début de upload...")
    file = request.files['file']
    platform = request.form.get('platform', 'générique')

    prompts = {
        "Instagram": (
            "Crée un résumé percutant pour une story Instagram, en français. Utilise des émojis 📸✨, "
            "des phrases courtes et visuelles. Le texte doit être prêt à être publié."
        ),
        "Facebook": "Crée un résumé informatif et engageant pour une publication Facebook, en français.",
        "Linkedin": "Fais un résumé très court et professionnel (moins de 280 caractères) pour Linkedin.",
        "Site web": "Crée un résumé clair, professionnel et structuré pour un site web.",
        "Presse": "Rédige un court communiqué de presse à partir de ce document.",
        "générique": "Fais un résumé court et clair de ce document PDF."
    }

    if file.filename.endswith('.pdf'):
        print("PDF reçu !")
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        print("Extraction du texte en cours...")
        text = extract_text_from_pdf(filepath)
        print("Chunking du texte en cours...")
        chunks = chunk_text(text)

        all_summaries = []
        print("Création du résumé en cours...")
        for chunk in chunks:
            prompt = f"{prompts.get(platform, prompts['générique'])}\n\nTexte :\n{chunk}\n\nRésumé :"
            summary = generate_summary(prompt)
            all_summaries.append(summary)
        print("Le résumé envoyé est :",all_summaries)
        print("Réponse envoyé au JSON !")
        return jsonify({"summary": "\n\n".join(all_summaries)})
    else:
        return "Format non pris en charge", 400

if __name__ == '__main__':
    app.run(debug=True)