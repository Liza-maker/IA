from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import fitz  # PyMuPDF
import requests
from flask_cors import CORS
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import ollama
from sentence_transformers import SentenceTransformer
import faiss
import time
import pdfplumber
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from PIL import Image
import easyocr
reader = easyocr.Reader(['fr'])

print("Début du programme...")
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://127.0.0.1:5000", "http://localhost:5001"]}})
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
OLLAMA_URL = "http://localhost:11434/api/generate"

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        page_text = page.get_text()
        if page_text.strip():
            text += page_text + "\n"
        else:
            # OCR avec EasyOCR
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            temp_image_path = "temp_image.png"
            img.save(temp_image_path)

            # Reconnaissance du texte dans l'image
            results = reader.readtext(temp_image_path, detail=0, paragraph=True)
            ocr_text = "\n".join(results)
            text += ocr_text + "\n"

            # Nettoyage du fichier temporaire
            os.remove(temp_image_path)
    return text

# Découpe simple en chunks
def chunk_text(text, max_words=500):
    words = text.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# Génération du résumé/communiqué via Ollama
def generate_summary(prompt):
    response = requests.post(OLLAMA_URL, json={
        "model": "llama3.1",
        "prompt": prompt,
        "temperature": 0.2,
        "stream": False
    })
    result = response.json()
    return result.get("response", "").strip()

def readPdfOLD(list_file):
    """
    Fonction qui transforme des pdf en texte :

    Args:
        file (list): liste des fichiers

    Returns:
        text : contenu du fichier
    """
    list_texts = []
    for file in list_file:
        reader = PdfReader(file)
        nb_pages = len(reader.pages)
        for i in range(nb_pages):
            page = reader.pages[i]
            text = page.extract_text()
            list_texts.append(text)
    return list_texts
#document = readPdf(["pdf-exemple.pdf","sample.pdf","rugby.pdf","mes-fiches-animaux-de-la-ferme.pdf","vaches.pdf"])

def readPdf(list_file):
    """
    Fonction qui transforme des pdf en texte (avec gestion des tableaux) :

    Args:
        list_file (list): liste des fichiers

    Returns:
        text : contenu du fichier
    """
    print("Initialisation du read pdf...")
    print(f"Document traité actuellement : {list_file}")
    list_texts = []
    for file in list_file:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                tables = page.extract_tables()
                for table in tables:
                    table_text = '\n'.join([
                        ' | '.join(cell if cell is not None else '' for cell in row)
                        for row in table if row
                    ])
                    text += "\n\nTABLE:\n" + table_text
                list_texts.append(text)
    print("Fin du read pdf...")
    return list_texts

def chuncking_doc(file):
    """Fonction qui créé des chunks sur le texte

    Args:
        doc (text): les textes des pdf

    Returns:
        chunks : textes pré-traités
    """
    print("Début du chunking...")
    print(f"Intérieur de la liste : {file}")
    #Fait une découpe plus intelligente que ma fonction grâce à chunk_overlap qui va essayer de trouver une phrase avant la découpe.
    docs = [Document(page_content="\n".join(pages)) for pages in file]
    splitter = RecursiveCharacterTextSplitter(chunk_size = 400, chunk_overlap=75,separators=["\n\n","\n",".","!","?"])
    splits_docs = splitter.split_documents(docs)
    chunks = [doc.page_content for doc in splits_docs]
    #On enlève tous les sauts de ligne
    def clean_chunk(text):
        return ' '.join(text.replace('\n', ' ').split())
    chunks = [clean_chunk(chunk) for chunk in chunks]
    print("Fin du chunk...")
    return chunks

# Embedding
def embedding_texts(chunks, user_text):
    if chunks:
        modelEmbedding = SentenceTransformer("all-MiniLM-L6-v2")
        embedding = modelEmbedding.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
        embedding_user = modelEmbedding.encode([user_text], convert_to_numpy=True).astype("float32")
        dimension = embedding.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embedding)
        distances, indices = index.search(embedding_user, k=4)
        return [chunks[idx] for idx in indices[0]]
    return False

# Génération de réponse Ollama
def predict_with_ollama(context, question, model_name="llama3.1"):
    prompt = f"""Voici des extraits de documents :\n{context}\n\nQuestion : {question}\n :
    Répond en utilisant principalement les extraits de document et en reformulant.
    N'inclus ni ton avis ni ton analyse.
    Si la question n'a pas de rapport répond : 'Désolé les documents ne répondent pas à votre question'.\n
    Réponse : """
    result = ollama.generate(model=model_name, prompt=prompt)
    return result["response"]

def generate_pdf(content, output_path):

    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4

    # Couleurs ASBH
    rouge_asbh = colors.HexColor("#ec2423")
    violet_asbh = colors.HexColor("#1d0e77")
    logo_path = "static/images/logo_asbh.png"

    # --- Logo ---
    if os.path.exists(logo_path):
        c.drawImage(logo_path, 40, height - 100, width=80, preserveAspectRatio=True, mask='auto')

    # --- Titre Principal sur fond rouge ---
    rect_x = 150
    rect_y = height #-30
    rect_width = 450
    rect_height = 100

    c.setFillColor(rouge_asbh)
    c.roundRect(rect_x, rect_y, rect_width, rect_height, 5, stroke=0, fill=1)

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 30)
    c.drawString(rect_x + 10, rect_y + 10, "Communiqué de Presse")

    # --- Corps du texte ---
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 12)
    left_margin = 50
    right_margin = 50
    top_margin = rect_y - 50
    line_height = 16

    # Découpe du texte en paragraphes
    paragraphs = content.split("\n\n")
    y_position = top_margin

    for para in paragraphs:
        lines = para.strip().split("\n")
        for line in lines:
            c.drawString(left_margin, y_position, line.strip())
            y_position -= line_height
            if y_position < 50:
                c.showPage()
                y_position = height - 50
                c.setFont("Helvetica", 12)
        y_position -= 8  # Espace entre paragraphes

    c.save()


#=================================================================
#               Début des app.route
#=================================================================
@app.route('/questionUtilisateur', methods=['POST'])
def questionUtilisateur():
    doc_stock = []
    files = request.files.getlist('file[]')
    print("=================\n File : ",files)
    text_user = request.form['question']
    if files and text_user:
        for file in files:
            docs = readPdf([file])
            doc_stock.append(docs)
        chunks = chuncking_doc(doc_stock)
        embeddings_text = embedding_texts(chunks,text_user)
        if not embeddings_text:
            return jsonify({"summary": "Désolé, le fichier n'est pas adapté à l'extraction de texte."})
        l = predict_with_ollama(embeddings_text,text_user)
        return jsonify({"summary": l})
    
@app.route('/upload', methods=['POST'])
def upload():
    start_time = time.time()
    files = request.files.getlist('files')
    platform = request.form.get('platform', 'générique')
    filter_style = request.form.get('style_filter', '')
    extra_prompt = request.form.get('custom_prompt', '').strip()

    base_prompts = {
        "Instagram": "Crée un résumé percutant pour une post Instagram, en français. Utilise des émojis 🔴🔵✨, des phrases courtes et visuelles. Le texte doit être prêt à être publié. Maximum 80 mots.",
        "Facebook": "Crée un résumé informatif et engageant pour une publication Facebook, en français. Ne rédige aucun commentaire ou note explicative. Le texte doit être prêt à être copié-collé tel quel. Maximum 200 mots. Ne génère qu’un seul résumé unique.",
        "Linkedin": "Fais un résumé très court et professionnel (moins de 280 caractères) pour un post de Linkedin.",
        "Site web": "Crée un résumé clair, professionnel et structuré pour un site web.",
        "Presse": "Rédige un communiqué de presse complet et structuré en français, prêt à être publié. Utilise un ton professionnel et dynamique. Mets en avant l’événement, les détails clés (date, lieu, organisateurs, contexte historique). "
        "Structure le texte avec des paragraphes. Attention : Rédige le texte de façon fluide et naturelle, évite les astérisques et les puces inutiles. Ne génère qu’un seul communiqué de presse de demi page A4 et unique.",
        "générique": "Fais un résumé court et clair de ce document PDF."
    }

    filter_prompts = {
        "attractif": "Rends ce résumé très attrayant et captivant.",
        "drôle": "Ajoute une touche d'humour et rends ce résumé drôle.",
        "créatif": "Sois créatif et original dans le résumé.",
        "professionnel": "Rends ce résumé très professionnel et sérieux."
    }

    # Extraire le texte de tous les fichiers PDF
    all_texts = []
    for file in files:
        if file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            text = extract_text_from_pdf(filepath)
            all_texts.append(text)

    if not all_texts:
        return "Aucun fichier PDF valide.", 400

    # Fusionner tout le texte en un seul string
    full_text = "\n\n".join(all_texts).strip()

    # Si texte court (< 3000 mots), on génère un seul résumé unique
    if len(full_text.split()) < 3000:
        prompt_parts = [base_prompts.get(platform, base_prompts['générique'])]
        if filter_style in filter_prompts:
            prompt_parts.append(filter_prompts[filter_style])
        if extra_prompt:
            prompt_parts.append(extra_prompt)
        prompt_parts.append(f"\nTexte :\n{full_text}\n\nCommuniqué :")
        full_prompt = "\n".join(prompt_parts)
        summary = generate_summary(full_prompt)
        summary = summary.replace('*', '')

        duration = round(time.time() - start_time, 2)
        return jsonify({"summary": summary, "duration": duration})

    else:
        # Texte long : découpe en chunks et résumé intermédiaire par chunk
        chunks = chunk_text(full_text, max_words=500)

        all_summaries = []
        for chunk in chunks:
            prompt_parts = [base_prompts.get(platform, base_prompts['générique'])]
            if filter_style in filter_prompts:
                prompt_parts.append(filter_prompts[filter_style])
            if extra_prompt:
                prompt_parts.append(extra_prompt)
            prompt_parts.append(f"\nTexte :\n{chunk}\n\nCommuniqué :")
            full_prompt = "\n".join(prompt_parts)
            summary = generate_summary(full_prompt)
            summary = summary.replace('*', '')
            all_summaries.append(summary)

        # Résumer les résumés intermédiaires en un résumé final unique
        final_prompt_parts = [base_prompts.get(platform, base_prompts['générique'])]
        if filter_style in filter_prompts:
            final_prompt_parts.append(filter_prompts[filter_style])
        if extra_prompt:
            final_prompt_parts.append(extra_prompt)
        final_prompt_parts.append("\nTexte :\n" + "\n\n".join(all_summaries) + "\n\nCommuniqué :")
        final_prompt = "\n".join(final_prompt_parts)
        final_summary = generate_summary(final_prompt)
        final_summary = final_summary.replace('*', '')

        duration = round(time.time() - start_time, 2)
        return jsonify({"summary": final_summary, "duration": duration})


@app.route('/generate_pdf', methods=['POST'])
def generate_pdf_route():
    content = request.json.get('content', '')
    output_pdf = os.path.join(UPLOAD_FOLDER, "communique_presse.pdf")
    generate_pdf(content, output_pdf)
    return send_file(output_pdf, mimetype='application/pdf')


if __name__ == '__main__':
    app.run(debug=True)
