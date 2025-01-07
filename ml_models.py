import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import sqlite3

# Connect to SQLite database
conn = sqlite3.connect('finance_assistant.db')
df = pd.read_sql_query("SELECT * FROM transactions", conn)
conn.close()

# Predict monthly expenses (regression)
df['month'] = pd.to_datetime(df['date']).dt.month
X = df[['month', 'amount']]
y = df['amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
regressor = RandomForestRegressor()
regressor.fit(X_train, y_train)
# Save the model (you can use pickle)
import pickle
with open('regressor.pkl', 'wb') as f:
    pickle.dump(regressor, f)

# Categorize expenses (classification)
X = df[['amount']]
y = df['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
# Save the model
with open('classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f)
