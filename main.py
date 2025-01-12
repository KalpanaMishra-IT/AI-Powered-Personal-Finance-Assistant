from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
import pandas as pd
import pickle
from transformers import pipeline
import Train, prediction

app = Flask(__name__)
app.secret_key = 'your_secret_key'


# Database connection
def get_db_connection():
    conn = sqlite3.connect('finance_assistant.db')
    conn.row_factory = sqlite3.Row
    return conn


# Initialize database tables
def init_db():
    conn = get_db_connection()
    with conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            username TEXT NOT NULL UNIQUE,
                            password TEXT NOT NULL
                        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS transactions (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER,
                            amount REAL,
                            category TEXT,
                            date TEXT,
                            FOREIGN KEY (user_id) REFERENCES users(id)
                        )''')
    conn.close()


# Load ML models
with open('regressor.pkl', 'rb') as f:
    regressor = pickle.load(f)
with open('classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

# NLP pipeline
nlp = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')


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
            return redirect(url_for('home'))
        return 'Invalid credentials'
    return render_template('login.html')


@app.route('/home')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM transactions WHERE user_id = ?", conn, params=(session['user_id'],))
    conn.close()

    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    categories = ['rent', 'food', 'entertainment', 'savings']
    budget_warning = None
    investment_suggestions = []
    monthly_expense=prediction.predictions_categories(df)

    if not df.empty:

        total_expense = sum(monthly_expense)
        budget_limit = 35000

        if total_expense > budget_limit:
            budget_warning = "You have exceeded your monthly budget! Consider cutting back on discretionary expenses."
            investment_suggestions=["No money to save","Try next month for saving, as expenses was greater then the budget_limit"]
        elif total_expense > 0.8 * budget_limit:
            budget_warning = "You are close to exceeding your budget. Monitor your spending."

            investment_suggestions=["Try to save more money for investment strategy","Use Fixed deposit to save"]

        if total_expense < 0.5 * budget_limit:
            investment_suggestions = [
                "Consider investing in mutual funds.",
                "Explore fixed deposits for secure savings.",
                "Look into SIPs for steady growth."
            ]

    else:
        monthly_expense = []

    combined_data = list(zip(monthly_expense, categories))
    if request.method == 'POST':
        query = request.form['query']
        response = nlp(question=query, context="User financial data analysis")
        combined_data = list(zip(monthly_expense, categories))
        return render_template('dashboard.html', response=response['answer'], monthly_expense=monthly_expense,
                                budget_warning=budget_warning,budget_limit=budget_limit,total_predicted_expenses=total_expense,
                               investment_suggestions=investment_suggestions,categories=categories,combined_data=combined_data)

    return render_template('dashboard.html', monthly_expense=monthly_expense,budget_limit=budget_limit,total_predicted_expenses=total_expense,
                           budget_warning=budget_warning, investment_suggestions=investment_suggestions,categories=categories,combined_data=combined_data)


@app.route('/add_transaction', methods=['GET', 'POST'])
def add_transaction():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        date = request.form['date']
        amount = request.form['amount']
        category = request.form['category']

        conn = get_db_connection()
        conn.execute('INSERT INTO transactions (user_id, amount, category, date) VALUES (?, ?, ?, ?)',
                     (session['user_id'], amount, category, date))
        conn.commit()
        conn.close()

        return redirect(url_for('add_transaction'))

    return render_template('add_transaction.html')

@app.route('/retrain_model', methods=['GET', 'POST'])
def retrain_model():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Fetch the latest transaction data
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM transactions WHERE user_id = ?", conn, params=(session['user_id'],))
    conn.close()

    if df.empty:
        return "No data available for retraining!"

    # Train a simple classification model
    status = Train.train_start(df)

    if status==True:
        return "Model trained successfully!"
    else:
        return "Model training Failed "

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
