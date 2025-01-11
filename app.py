from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import pandas as pd
import pickle
from transformers import pipeline
import logging

# Initialize app and logging
app = Flask(__name__)
app.secret_key = 'your_secret_key'
logging.basicConfig(level=logging.DEBUG)

# Load models
try:
    with open('regressor.pkl', 'rb') as f:
        regressor = pickle.load(f)
    with open('classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)
except FileNotFoundError as e:
    logging.error(f"Model file not found: {e}")
    regressor = None
    classifier = None

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
        with get_db_connection() as conn:
            conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            conn.commit()
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        with get_db_connection() as conn:
            user = conn.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password)).fetchone()
        if user:
            session['user_id'] = user['id']
            flash('Login successful!')
            return redirect(url_for('dashboard'))
        flash('Invalid credentials. Please try again.')
    return render_template('login.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user_id' not in session:
        flash('Please log in to access the dashboard.')
        return redirect(url_for('login'))

    if request.method == 'POST':
        query = request.form['query']
        if query:
            response = nlp(query)
            return render_template('dashboard.html', response=response[0]['label'])
        flash('Please enter a valid query.')

    # Retrieve transactions
    with get_db_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM transactions", conn)

    if df.empty:
        flash('No transaction data available.')
        return render_template('dashboard.html', monthly_expense=None, expense_category=None)

    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month

    try:
        # Predict monthly expenses
        monthly_expense = regressor.predict(df[['month', 'amount']]) if regressor else None
        # Categorize expenses
        expense_category = classifier.predict(df[['amount']]) if classifier else None
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        monthly_expense = None
        expense_category = None

    return render_template(
        'dashboard.html',
        monthly_expense=monthly_expense.tolist() if monthly_expense is not None else None,
        expense_category=expense_category.tolist() if expense_category is not None else None
    )

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
