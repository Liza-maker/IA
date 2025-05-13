import fitz  # PyMuPDF
#Chargement du modèle de résumé T5-small
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

#extraire un texte
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

#Découpage du texte en petits morceaux
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

#summariser un textee
def summarize_text(text_chunks, model, tokenizer):
    summaries = []
    for chunk in text_chunks:
        inputs = tokenizer("summarize: " + chunk, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(inputs.input_ids, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return " ".join(summaries)

def summarize_pdf(pdf_path):
    print("📄 Extraction du texte du PDF...")
    text = extract_text_from_pdf(pdf_path)
    
    print("✂️ Découpage en morceaux...")
    text_chunks = chunk_text(text)
    
    print("🤖 Chargement du modèle T5-small...")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    
    print("📝 Génération du résumé...")
    summary = summarize_text(text_chunks, model, tokenizer)
    
    return summary

# utilisation
if __name__ == "__main__":
    pdf_path = "pdf-exemple.pdf"  
    result = summarize_pdf(pdf_path)
    print("\n Résumé :\n")
    print(result)
