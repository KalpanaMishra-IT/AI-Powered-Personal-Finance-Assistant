import sqlite3

from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

def predictions_categories(df):
    categories = ['rent', 'food', 'entertainment', 'savings']
    seq_length=2
    predictions=[]
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month

    for category in categories:
        loaded_model = load_model(f'lstm_{category}_model.h5')
        cat_df = df[df['category'].str.lower() == category].groupby('month')['amount'].sum().reset_index()
        # Use the loaded model for prediction
        last_sequence = cat_df.iloc[-seq_length:,-1:]
        test_input = np.array(last_sequence).reshape((1, seq_length, 1))
        predicted = loaded_model.predict(test_input)
        predictions.append(int(predicted[0][0]))
        print("category-->", category)
        print(f"Predicted next number: {predicted[0][0]:.2f}")

    return predictions


# conn = sqlite3.connect('finance_assistant.db')
# conn.row_factory = sqlite3.Row
# df = pd.read_sql_query("SELECT * FROM transactions WHERE user_id = ?", conn, params=(1,))
# conn.close()
#
# df['date'] = pd.to_datetime(df['date'])
# df['month'] = df['date'].dt.month
# ans=predictions_categories(df)
# print(ans)
# print(sum(ans))