<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ScanSport</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
  <style>
    body {
      margin: 0;
      font-family: 'Inter', sans-serif;
      background-color: #ffffff;
      color: #333;
    }

    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 20px 40px;
      background: linear-gradient(90deg, #f94d6a, #7b2ff7);
      color: white;
    }

    .logo-title {
      font-size: 28px;
      font-weight: 600;
    }

    .container {
      display: flex;
      justify-content: space-between;
      padding: 40px;
      gap: 40px;
    }

    .left-column {
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    .upload-box, .footer {
      border-radius: 12px;
      padding: 30px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
      background-color: #f7f7f7;
    }

    .upload-box {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
    }

    .upload-box form {
      width: 100%;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .upload-box input[type="file"] {
      margin: 20px 0;
      flex: 1;
    }

    .upload-box button {
      margin-top: auto;
      padding: 10px 20px;
      background-color: #007b2f;
      color: white;
      border: none;
      border-radius: 6px;
      cursor: pointer;
    }

    .summary-box {
      flex: 1;
      border: 2px solid #d6336c;
      border-radius: 12px;
      padding: 30px;
      background-color: #fff;
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
      display: flex;
      flex-direction: column;
      justify-content: flex-end;
      align-items: flex-end;
    }

    .summary-box h3 {
      align-self: flex-start;
      margin-top: 0;
      color: #d6336c;
    }

    .summary-box p {
      align-self: flex-start;
    }

    .summary-box button {
      margin-top: auto;
      padding: 8px 16px;
      background-color: #f8f9fa;
      border: 1px solid #ccc;
      border-radius: 6px;
      cursor: pointer;
      align-self: flex-end;
    }

    .footer {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .footer-input-wrapper {
      display: flex;
      align-items: center;
      border: 1px solid #ccc;
      border-radius: 6px;
      background-color: #f0f0f0;
      padding: 5px 10px;
    }

    .footer input[type="text"] {
      flex: 1;
      padding: 10px;
      border: none;
      background: transparent;
      outline: none;
    }

    .footer button {
      padding: 10px;
      border-radius: 6px;
      border: none;
      background-color: transparent;
      color: #555;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 20px;
      cursor: pointer;
    }


    .socials {
      margin-top: 15px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      justify-content: center;
    }
    .footer-input-wrapper button::before {
      content: '\27A4'; /* flèche */
    }

    .socials button {
      padding: 10px 20px;
      border: 1px solid #333;
      border-radius: 20px;
      background-color: white;
      cursor: pointer;
      font-weight: bold;
    }
  </style>
</head>
<body>
  <header>
    <div class="logo-title">ScanSport</div>
    <img src="images/logo_asbh.png" alt="ASBH Logo" height="90">
  </header>

  <div class="container">
    <div class="left-column">
      <div class="upload-box">
        <img src="images/Upload-Icon.png" alt="Upload" width="80">
        <p>Déposer votre fichier (.mp4, .pdf) ici</p>
        <form action="upload.php" method="POST" enctype="multipart/form-data">
          <input type="file" name="file" accept=".mp4,.pdf" required>
          <button type="submit">Confirmer</button>
        </form>
      </div>

      <div class="footer">
        <div class="footer-input-wrapper">
          <input type="text" placeholder="Demander quelque chose....">
          <button title="Envoyer" aria-label="Envoyer"></button>
        </div>
        <div class="socials">
          <button>Instagram</button>
          <button>Site web</button>
          <button>Facebook</button>
          <button>Twitter/X</button>
        </div>
      </div>
    </div>

    <div class="summary-box">
      <h3>Génération du résumé...</h3>
      <p id="summary-text">En attente du fichier...</p>
      <button onclick="location.reload()">Réessayez</button>
    </div>
  </div>
  <script>
  async function envoyerMessage() {
    const userMessage = document.querySelector('.footer input').value;

    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer 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" // Mets ta clé ici
      },
      body: JSON.stringify({
        model: "gpt-3.5-turbo",
        messages: [
          { role: "system", content: "Tu es un assistant sportif qui peut résumer des textes ou répondre à des questions." },
          { role: "user", content: userMessage }
        ]
      })
    });

    const data = await response.json();
    const botReply = data.choices[0].message.content;

    document.getElementById("summary-text").innerText = botReply;
  }

  document.querySelector(".footer button").addEventListener("click", envoyerMessage);
</script>

</body>
</html>