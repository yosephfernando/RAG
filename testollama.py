import ollama

# Use the correct model name
response = ollama.chat(model="llama3.2:latest", messages=[{"role": "user", "content": "Hello, how are you?"}])

# Print the response
print(response)
