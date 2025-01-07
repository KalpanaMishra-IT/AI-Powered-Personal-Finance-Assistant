import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import plotly.express as px

# Connect to SQLite database
conn = sqlite3.connect('finance_assistant.db')
df = pd.read_sql_query("SELECT * FROM transactions", conn)
conn.close()

# Matplotlib visualization
df.groupby('category')['amount'].sum().plot(kind='bar')
plt.title('Expenses by Category')
plt.xlabel('Category')
plt.ylabel('Total Amount')
plt.show()

# Plotly visualization
fig = px.pie(df, names='category', values='amount', title='Expenses Distribution')
fig.show()
