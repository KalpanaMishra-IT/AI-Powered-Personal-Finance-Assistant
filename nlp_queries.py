from transformers import pipeline

# Load sentiment-analysis pipeline
nlp = pipeline('sentiment-analysis')

# Example query
query = "How much did I spend on groceries last week?"

# Process query (this is a simplified example)
print(nlp(query))
