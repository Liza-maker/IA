start /min pip install -r requirements.txt
CD templates/
start /min cmd.exe /k "cd .. && python api.py"
start /min cmd.exe /k "ollama run llama3.1"

start /min python -m http.server 5001


