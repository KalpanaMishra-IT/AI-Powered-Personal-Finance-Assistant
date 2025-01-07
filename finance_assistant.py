import pandas as pd
import sqlite3
import pickle
from transformers import pipeline
import matplotlib.pyplot as plt
import plotly.express as px

# Load models
with open('regressor.pkl', 'rb') as f:
    regressor = pickle.load(f)
with open('classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

# Connect to SQLite database
conn = sqlite3.connect('finance_assistant.db')
df = pd.read_sql_query("SELECT * FROM transactions", conn)
conn.close()

# Predict monthly expenses
df['month'] = pd.to_datetime(df['date']).dt.month
monthly_expense = regressor.predict(df[['month', 'amount']])
print("Predicted monthly expense:", monthly_expense)

# Categorize expenses
expense_category = classifier.predict(df[['amount']])
print("Expense categories:", expense_category)

# NLP query processing
nlp = pipeline('sentiment-analysis')
query = "How much did I spend on groceries last week?"
print(nlp(query))

# Data visualization
df.groupby('category')['amount'].sum().plot(kind='bar')
plt.title('Expenses by Category')
plt.show()

fig = px.pie(df, names='category', values='amount', title='Expenses Distribution')
fig.show()