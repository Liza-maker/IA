from flask import Flask, request, jsonify
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

print("Début du programme...")
app = Flask(__name__)
#Autoriser la redirection vers les liens adresses suivantes :
CORS(app, resources={r"/*": {"origins": ["http://127.0.0.1:5000", "http://localhost:5001"]}})
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
        "temperature":0.2,
        "stream": False
    })
    result = response.json()
    return result.get("response", "").strip()

def readPdf(list_file):
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

def chuncking_doc(file):
    """Fonction qui créé des chunks sur le texte

    Args:
        doc (text): les textes des pdf

    Returns:
        chunks : textes pré-traités
    """
    #Fait une découpe plus intelligente que ma fonction grâce à chunk_overlap qui va essayer de trouver une phrase avant la découpe.
    docs = [Document(page_content=page_text) for page_text in file]
    splitter = RecursiveCharacterTextSplitter(chunk_size = 400, chunk_overlap=75,separators=["\n\n","\n",".","!","?"])
    splits_docs = splitter.split_documents(docs)
    chunks = [doc.page_content for doc in splits_docs]
    #On enlève tous les sauts de ligne
    def clean_chunk(text):
        return ' '.join(text.replace('\n', ' ').split())
    chunks = [clean_chunk(chunk) for chunk in chunks]
    return chunks


def embedding_texts(chunks,user_text):
    """Fonction qui fait des embeddings sur les chunks

    Args:
        chunks (list): liste des chunks
        user_text (str): question de l'utilisateur

    Returns:
        output_text : Top k (k=4) des similarités entre chunks et question
    """
    print("Chunks : ",chunks)
    if len(chunks) > 0:
        
        #On embedding (transférer les phrases par des vecteurs) (documents puis question utilisateur)
        modelEmbedding = SentenceTransformer("all-MiniLM-L6-v2")
        embedding = modelEmbedding.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

        embedding_user = modelEmbedding.encode([user_text],show_progress_bar=True, convert_to_numpy=True).astype("float32")    
        dimension = embedding.shape[1]
        index = faiss.IndexFlatIP(dimension)  # 32 = nombre de voisins (standard)
        output_text = []
        #on ajoute les embeddings : 
        index.add(embedding)
        distances, indices = index.search(embedding_user, k=4)
        for i, idx in enumerate(indices[0]):
            output_text.append(chunks[idx])
        return output_text
    return False

def predict_with_ollama(context, question, model_name="llama3.1"):
    """Fonction qui génère le résultat du modèle

    Args:
        context (list)  : chunks les + similaires à la question
        question (str)  : question de l'utilisateur
        model_name (str): modèle utilisé

    Returns:
        result["response"] : Réponse du modèle à la question de l'utilisateur
    """
    print(context)
    prompt = f"""Voici des extraits de documents :\n{context}\n\nQuestion : {question}\n :
    Répond en utilisant principalement les extraits de document et en reformulant.
    N'inclus ni ton avis ni ton analyse.
    Si la question n'a pas de rapport répond : 'Désolé les documents ne répondent pas à votre question'.\n
    Réponse : """
    result = ollama.generate(model=model_name, prompt= prompt)
    return result["response"]

#=================================================================
#               Début des app.route
#=================================================================
@app.route('/questionUtilisateur', methods=['POST'])
def questionUtilisateur():
    file = request.files['file']
    text_user = request.form['question']
    if file and text_user:
            docs = readPdf([file])
            chunks = chuncking_doc(docs)
            embeddings_text = embedding_texts(chunks,text_user)
            if not embeddings_text:
                return jsonify({"summary": "Désolé, le fichier n'est pas adapté à l'extraction de texte."})
            l = predict_with_ollama(embeddings_text,text_user)
            return jsonify({"summary": l})



@app.route('/upload', methods=['POST'])
def upload():
    start_time = time.time()

    file = request.files['file']
    platform = request.form.get('platform', 'générique')
    filter_style = request.form.get('style_filter', '')
    extra_prompt = request.form.get('custom_prompt', '').strip()
    print("Informations reçus !")
    base_prompts = {
        "Instagram": "Crée un résumé percutant pour une story Instagram, en français. Utilise des émojis 📸✨, des phrases courtes et visuelles. Le texte doit être prêt à être publié.",
        "Facebook": "Crée un résumé informatif et engageant pour une publication Facebook, en français.",
        "Linkedin": "Fais un résumé très court et professionnel (moins de 280 caractères) pour Linkedin.",
        "Site web": "Crée un résumé clair, professionnel et structuré pour un site web.",
        "Presse": "Rédige un court communiqué de presse à partir de ce document.",
        "générique": "Fais un résumé court et clair de ce document PDF."
    }

    filter_prompts = {
        "attrayant": "Rends ce résumé très attrayant et captivant.",
        "drôle": "Ajoute une touche d'humour et rends ce résumé drôle.",
        "créatif": "Sois créatif et original dans le résumé.",
        "professionnel": "Rends ce résumé très professionnel et sérieux."
    }

    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        text = extract_text_from_pdf(filepath)
        chunks = chunk_text(text)
        print("Chunks effectués pour résumer !")
        all_summaries = []
        for chunk in chunks:
            prompt_parts = [base_prompts.get(platform, base_prompts['générique'])]
            if filter_style in filter_prompts:
                prompt_parts.append(filter_prompts[filter_style])
            if extra_prompt:
                prompt_parts.append(extra_prompt)
            prompt_parts.append(f"\nTexte :\n{chunk}\n\nRésumé :")

            full_prompt = "\n".join(prompt_parts)
            summary = generate_summary(full_prompt)
            all_summaries.append(summary)

        duration = round(time.time() - start_time, 2)
        return jsonify({"summary": "\n\n".join(all_summaries), "duration": duration})
    else:
        return "Format non pris en charge", 400





if __name__ == '__main__':
    app.run(debug=True)