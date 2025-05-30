from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialisation du modèle une seule fois
MODEL_NAME = "llama3.1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, max_tokens=512):
    sentences = text.split('. ')
    chunks, chunk = [], ""
    for sentence in sentences:
        if len(chunk) + len(sentence.split()) < max_tokens:
            chunk += sentence + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + ". "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
def summarize_text(text_chunks):
    summaries = []
    for chunk in text_chunks:
        # 1) Formate un prompt clair pour un modèle instruct
        prompt = (
            "Créer un résumé court en français du fichier suivant:\n\n"
            f"{chunk}\n\n"
            "Résumé:"
        )

        # 2) Tokenize et pousse sur le GPU
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(device)

        # 3) Génère la réponse
        outputs = model.generate(
            **inputs,
            do_sample=True,           # active l'échantillonnage
            temperature=0.5,          # contrôle de la créativité
            top_p=0.9,                # filtrage nucleus
            max_new_tokens=100,       # longueur max du résumé
            pad_token_id=tokenizer.eos_token_id
        )

        # 4) Decode en texte brut
        summary = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        # On retire la partie du prompt pour ne garder que le résumé
        summary = summary.split("Summary:")[-1].strip()

        summaries.append(summary)

    # Concatène les résumés par double saut de ligne
    return "\n\n".join(summaries)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        text = extract_text_from_pdf(filepath)
        chunks = chunk_text(text)
        summary = summarize_text(chunks)

        return jsonify({"summary": summary})
    else:
        return "Format non pris en charge", 400

    
@app.route("/")
def index():
    return render_template("scansport.html")

if __name__ == '__main__':
    app.run()
