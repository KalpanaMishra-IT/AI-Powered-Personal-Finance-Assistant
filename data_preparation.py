import pandas as pd
import sqlite3

# Example transaction data
data = {'date': ['2025-01-01', '2025-01-02', '2025-01-03'],
        'category': ['Groceries', 'Entertainment', 'Rent'],
        'amount': [50, 20, 500]}
df = pd.DataFrame(data)

# Connect to SQLite database
conn = sqlite3.connect('finance_assistant.db')
cursor = conn.cursor()

# Create table for storing transactions
cursor.execute('''CREATE TABLE IF NOT EXISTS transactions 
                  (id INTEGER PRIMARY KEY, date TEXT, category TEXT, amount REAL)''')
conn.commit()

# Create table for storing users
cursor.execute('''CREATE TABLE IF NOT EXISTS users
                  (id INTEGER PRIMARY KEY, username TEXT, password TEXT)''')
conn.commit()

# Insert sample data
df.to_sql('transactions', conn, if_exists='replace', index=False)

# Close connection
conn.close()