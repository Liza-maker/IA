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

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Flowable, PageTemplate, BaseDocTemplate, Frame
from reportlab.lib.units import cm

from reportlab.lib.enums import TA_JUSTIFY

pdfmetrics.registerFont(TTFont('Anton', 'static/Anton-Regular.ttf'))


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



class HeaderWithBackground(Flowable):
    def __init__(self, logo_path=None, title_text="Communiqué de presse"):
        Flowable.__init__(self)
        self.logo_path = logo_path
        self.title_text = title_text
        self.height = 3.5 * cm
        self.page_width = A4[0]

    def draw(self):
        c = self.canv
        x = 0
        y = 0

        logo_size = 3 * cm
        offset_x = 0

        logo_bg_width = logo_size + 1 * cm
        c.setFillColor(colors.HexColor("#f2f2f2"))
        c.rect(x, y, logo_bg_width, self.height, stroke=0, fill=1)

        if self.logo_path and os.path.exists(self.logo_path):
            c.drawImage(
                self.logo_path,
                x + 0.5 * cm, y + (self.height - logo_size) / 2,
                width=logo_size, height=logo_size,
                preserveAspectRatio=True, mask='auto'
            )
        offset_x = logo_bg_width

        rect_width = self.page_width - offset_x
        c.setFillColor(colors.HexColor("#ec2423"))
        c.rect(offset_x, y, rect_width, self.height, stroke=0, fill=1)

        c.setFillColor(colors.white)
        c.setFont("Anton", 35)
        text_x = offset_x + 1 * cm
        text_y = y + self.height / 2 - 10
        c.drawString(text_x, text_y, self.title_text)

class FooterGraphics(Flowable):
    def __init__(self):
        Flowable.__init__(self)
        self.page_width, self.page_height = A4

    def draw(self):
        c = self.canv
        x_right = self.page_width
        y_bottom = 0

        # Rectangle bleu
        c.setFillColor(colors.HexColor("#1d0e77"))
        c.saveState()
        c.translate(x_right - 5*cm, y_bottom)
        c.rotate(-45)
        c.rect(0, 0, 2*cm, 8*cm, stroke=0, fill=1)
        c.restoreState()

        # Rectangle rouge — on le place plus à droite et plus haut
        c.setFillColor(colors.HexColor("#ec2423"))
        c.saveState()
        c.translate(x_right - 4*cm, y_bottom + 4*cm)
        c.rotate(-45)
        c.rect(0, 0, 1*cm, 3*cm, stroke=0, fill=1)
        c.restoreState()

def draw_footer(canvas, doc):
    footer = FooterGraphics()
    footer.canv = canvas
    canvas.saveState()
    canvas.translate(0, 0)
    footer.draw()
    canvas.restoreState()

def generate_pdf(content, output_path):
    doc = BaseDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=1.5*cm, leftMargin=1.5*cm,
        topMargin=0*cm, bottomMargin=2*cm
    )

    frame = Frame(doc.leftMargin, doc.bottomMargin, 
                  doc.width, doc.height, id='normal')
    template = PageTemplate(id='with_footer', frames=frame, onPage=draw_footer)
    doc.addPageTemplates([template])

    styles = getSampleStyleSheet()
    normal_style = ParagraphStyle(
        'NormalStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=13,
        leading=14,
        spaceAfter=9,
        alignment=TA_JUSTIFY  # Plus d'espace entre paragraphes
    )

    elements = []

    logo_path = "static/images/logo_asbh.png"
    elements.append(HeaderWithBackground(logo_path))
    elements.append(Spacer(1, 40))  # Plus d'espace après le header

    paragraphs = content.strip().split('\n\n')
    for para in paragraphs:
        para = para.strip().replace('\n', ' ')
        elements.append(Paragraph(para, normal_style))
        elements.append(Spacer(1, 12))

    doc.build(elements)

# Exemple d'utilisation
# generate_pdf("Ton texte ici", "output.pdf")



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
        "Instagram": "Crée un résumé percutant pour une post Instagram, en français. Utilise des émojis 🔴🔵✨, des phrases courtes et visuelles. Le texte doit être prêt à être publié. Maximum 80 mots. Termine un post avec les hashtags. ",
        "Facebook": "Crée un résumé informatif et engageant pour une publication Facebook, en français. Ne rédige aucun commentaire ou note explicative. Le texte doit être prêt à être copié-collé tel quel. Maximum 200 mots. Ne génère qu’un seul résumé unique.",
        "Linkedin": "Fais un résumé très court et professionnel (moins de 280 caractères) pour un post de Linkedin.",
        "Site web": "Crée un résumé clair, professionnel et structuré pour un site web.",
        "Presse": "Rédige un communiqué de presse complet et structuré en français, prêt à être publié. Utilise un ton professionnel et dynamique. Mets en avant l’événement et les détails clés. "
        "Rédige le texte sous forme fluide et naturelle, divisé en paragraphes, sans titre et sans notes entre crochets ou éléments à compléter. Le texte doit avoir une longueur équivalente à une demi-page A4 et être unique. ",
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
