from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import pandas as pd

# Sample Data (Replace with actual labeled data)
data = {
    'amount': [500, 2000, 5000, 7000, 10000, 15000],
    'category': ['Groceries', 'Utilities', 'Rent', 'Rent', 'Luxury', 'Luxury']
}

# Convert categories to numerical labels
df = pd.DataFrame(data)
df['category_label'] = df['category'].astype('category').cat.codes

# Features and Target
X = df[['amount']]
y = df['category_label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Classifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Save the model
with open('classifier.pkl', 'wb') as file:
    pickle.dump(classifier, file)

print("Classifier model trained and saved as classifier.pkl")
