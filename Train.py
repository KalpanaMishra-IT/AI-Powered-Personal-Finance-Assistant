import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import sqlite3


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def train_start(df):
    # Preprocessing: Prepare features and labels for training
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month

    categories = ['rent', 'food', 'entertainment', 'savings']

    for category in categories:
        cat_df = df[df['category'].str.lower() == category].groupby('month')['amount'].sum().reset_index()
        if cat_df.empty or len(cat_df) < 3:
            continue
        print(cat_df['amount'])
        data = cat_df['amount']

        seq_length = 2  # Use the past 5 numbers to predict the next
        X, y = create_sequences(data, seq_length)

        # Reshape X to fit LSTM input: [samples, time_steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Build the LSTM model
        model = Sequential([LSTM(50, activation='relu', input_shape=(seq_length, 1)),
                            Dense(1)  # Output layer
                            ])

        # Compile the model
        model.compile(optimizer='adam', loss='mse')

        # Train the model
        model.fit(X, y, epochs=200, verbose=0)
        model.save(f'lstm_{category}_model.h5')

    return True
# Initialize database tables

# conn = sqlite3.connect('finance_assistant.db')
# df = pd.read_sql_query("SELECT * FROM transactions WHERE user_id = ?", conn, params=(1,))
# conn.close()
# x=train_start(df)
# print(x)