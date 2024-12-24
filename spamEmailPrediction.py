import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the data
df = pd.read_csv('encoded-spam.csv', usecols=['v1', 'v2'])
df.columns = ['label', 'text']

# Data cleaning
df.dropna(inplace=True)

# Text preprocessing

# Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label'].apply(lambda x: 1 if x == 'spam' else 0)

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Precision: {precision_score(y_test, y_pred)}')
print(f'Recall: {recall_score(y_test, y_pred)}')
print(f'F1 Score: {f1_score(y_test, y_pred)}')
print(f'ROC AUC: {roc_auc_score(y_test, y_pred)}')

# Predict if a new email is spam or not
new_email = ["SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info"]
new_email_transformed = vectorizer.transform(new_email)
prediction = model.predict(new_email_transformed)
print('Spam' if prediction[0] == 1 else 'Not Spam')