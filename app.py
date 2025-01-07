from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
import pandas as pd
import pickle
from transformers import pipeline

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load models
with open('regressor.pkl', 'rb') as f:
    regressor = pickle.load(f)
with open('classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

# NLP pipeline
nlp = pipeline('sentiment-analysis', model='distilbert/distilbert-base-uncased-finetuned-sst-2-english')

# Database setup
def get_db_connection():
    conn = sqlite3.connect('finance_assistant.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
        conn.commit()
        conn.close()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password)).fetchone()
        conn.close()
        if user:
            session['user_id'] = user['id']
            return redirect(url_for('dashboard'))
        return 'Invalid credentials'
    return render_template('login.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        query = request.form['query']
        response = nlp(query)
        return render_template('dashboard.html', response=response[0]['label'])

    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM transactions", conn)
    conn.close()

    # Ensure 'month' column is added to the DataFrame
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month

    print(df)  # Debug: Print the DataFrame to check its content

    if not df.empty:
        # Predict monthly expenses
        monthly_expense = regressor.predict(df[['month', 'amount']])
        print("Predicted Monthly Expense:", monthly_expense)  # Debug: Print predictions

        # Categorize expenses
        expense_category = classifier.predict(df[['amount']])
        print("Expense Categories:", expense_category)  # Debug: Print categories

        return render_template('dashboard.html', monthly_expense=monthly_expense, expense_category=expense_category)
    else:
        return render_template('dashboard.html', response="No data available")

if __name__ == '__main__':
    app.run(debug=True)