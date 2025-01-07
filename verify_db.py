import sqlite3

# Connect to SQLite database
conn = sqlite3.connect('finance_assistant.db')
cursor = conn.cursor()

# Verify table existence and content
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users';")
table_exists = cursor.fetchone()
if table_exists:
    print("Table 'users' exists.")
else:
    print("Table 'users' does not exist.")

# Close connection
conn.close()