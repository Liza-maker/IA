from flask import Flask, request, jsonify
from flask_cors import CORS

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import ollama
from sentence_transformers import SentenceTransformer
import faiss


print("Début du programme...")
app = Flask(__name__)
#Autoriser la redirection vers les liens adresses suivantes :
CORS(app, resources={r"/*": {"origins": ["http://127.0.0.1:5000", "http://localhost:5001"]}})
OLLAMA_URL = "http://localhost:11434/api/generate"

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
    """On chunks le text

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
    #On embedding (transférer les phrases par des vecteurs) (documents puis question utilisateur)
    modelEmbedding = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = modelEmbedding.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

    embedding_user = modelEmbedding.encode([user_text],show_progress_bar=True, convert_to_numpy=True).astype("float32")
    #------------------------------------
    
    dimension = embedding.shape[1]
    print("Les dimensions sont : ",dimension)
    index = faiss.IndexFlatIP(dimension)  # 32 = nombre de voisins (standard)
    output_text = []
    #on ajoute les embeddings : 
    index.add(embedding)
    distances, indices = index.search(embedding_user, k=4)
    for i, idx in enumerate(indices[0]):
        output_text.append(chunks[idx])

    return output_text

def predict_with_ollama(context, question, model_name="llama3.1"):
    prompt = f"""Voici des extraits de documents :\n{context}\n\nQuestion : {question}\n :
    Répond en utilisant principalement les extraits de document et en reformulant.
    N'inclus ni ton avis ni ton analyse.
    Si la question n'a pas de rapport répond : 'Désolé les documents ne répondent pas à votre question'.\n
    Réponse : """
    result = ollama.generate(model=model_name, prompt= prompt)
    return result["response"]

    
@app.route('/questionUtilisateur', methods=['POST'])
def questionUtilisateur():
    file = request.files['file']
    text_user = request.form['question']
    if file and text_user:
            docs = readPdf([file])
            chunks = chuncking_doc(docs)
            embeddings_text = embedding_texts(chunks,text_user)
            l = predict_with_ollama(embeddings_text,text_user)
            return jsonify({"summary": l})
if __name__ == '__main__':
    app.run(debug=True)
