import pandas as pd
import nltk
import joblib
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re

nltk.download('stopwords')

# Load dataset
data = pd.read_csv("final_dataset.csv")

# Improved cleaning function (URL-safe)
def clean_text(text):
    text = str(text).lower()
    
    # If it's a URL, keep structure
    if text.startswith("http"):
        return text
    
    # Normal message cleaning
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

# Apply cleaning
data['text'] = data['text'].apply(clean_text)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['text'])
y = data['label']

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X, y)

# Save
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained successfully!")
print("Dataset size:", len(data))

