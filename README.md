### INSTALL DEPENDENCIES ###
`pip install -r requirements.txt`

Make sure enable your virtual env first  before execute the script

### SET PINECONE APIKEY IN .env FILE ###
PINECONE_API_KEY=YOUR_API_KEY

### MAKE DOCUMEN(PDF) USING THIS SCRIPT ###
`python Pdfingestion.py`

### INTERACT WITH THE LLM USING THIS SCRIPT ###
`python main.py`

### CHANGE QUERY ###
You can modify line 63 in main.py. Example:
```
query = "What is Proximity graph techniques ?"
```

### NOTES ###
- Make sure you already install ollama3.2:latest on your machine. link to download https://ollama.com/download
- To test your ollama working, use script: `python testollama.py`