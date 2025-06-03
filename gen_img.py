import torch
import clip 
from PIL import Image
import csv 
import matplotlib.pyplot as plt 
import os 

device = "cpu"

model, preprocess = clip.load("ViT-L/14@336px", device=device)
print('CLIP est ok')

image_path = "images/plaquage4.jpg"

image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

themes = [
    "Spectateurs de rugby",
    "Plaquage au rugby",
    "Fair-play au rugby",
    "Arbitre de rugby",
    "Joueur de rugby",
    "Action de rugby",
    "Mascotte de rugby"
]

extended_themes = ["Supporters en tribune lors d'un match de rugby",
    "Joueur plaquant un adversaire au rugby",
    "Poignée de main entre deux joueurs après un match de rugby",
    "Arbitre en action sur un terrain de rugby",
    "Joueur courant avec un ballon de rugby",
    "Mêlée de rugby intense",
    "Mascotte du stade pendant un match de rugby"
]
text_inputs = clip.tokenize(themes).to(device)
text_inputs_extended = clip.tokenize(extended_themes).to(device)

with torch.no_grad(): 
    image_features = model.encode_image(image)
    
    text_features = model.encode_text(text_inputs)
    text_features2= model.encode_text(text_inputs_extended)
    
    similarity = (image_features @ text_features.T).softmax(dim=-1)
    similarity2= (image_features @ text_features2.T).softmax(dim=-1)
    # 🔍 Récupérer les scores et indices triés
    sorted_indices = similarity.squeeze().argsort(descending=True)  # Trie les indices du plus élevé au plus bas
    sorted_indices_extended = similarity2.squeeze().argsort(descending=True)
    sorted_scores = similarity.squeeze()[sorted_indices]  # Trie les scores correspondants
    sorted_scores_extended = similarity2.squeeze()[sorted_indices_extended]


best_match_index = similarity.argmax().item()
best_theme = themes[best_match_index]
best_score = similarity.max().item()

best_match_index_extended = similarity2.argmax().item()
best_theme_extended = extended_themes[best_match_index_extended]
best_score_extended = similarity2.max().item()

print(f"🔹 L'image correspond le plus à : {themes[best_match_index]}")
print(f"🎯 Score de correspondance : {similarity.max().item():.4f}")

print(f"🔹 L'image correspond le plus à : { extended_themes[best_match_index_extended]}")
print(f"🎯 Score de correspondance : {best_score_extended:.4f}")


separator = "------" 
with open("test.csv", mode="a", encoding="utf-8", newline="") as f: 
    writer = csv.writer(f)
    writer.writerow([separator, separator, separator])
    writer.writerow(["Image analysée", image_path])
    writer.writerow([separator, separator, separator])
    writer.writerow(["Thème correspondant", best_theme])
    writer.writerow(["Score de correspondance", best_score])
    writer.writerow([separator, separator, separator])
    writer.writerow([separator, separator, separator])
    writer.writerow(["Thème correspondant pour les descriptions", best_theme_extended])
    writer.writerow(["Score de correspondance", best_score_extended])


    

