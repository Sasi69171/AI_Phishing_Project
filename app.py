from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    accuracy = None
    risk = None
    message = ""

    if request.method == 'POST':
        text = request.form.get('text', '')

        # ⚠️ IMPORTANT: Do NOT clean again — use raw text
        vectorized = vectorizer.transform([text])

        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized).max()
        accuracy = round(confidence * 100, 2)

        if prediction == 1:
            result = "Phishing"
            risk = "High"
            message = "⚠️ This message shows characteristics of a phishing attack. Please do not trust it."
        else:
            result = "Legitimate"
            risk = "Low"
            message = "✅ This message appears safe and legitimate."

    return render_template(
        'index.html',
        result=result,
        accuracy=accuracy,
        risk=risk,
        message=message
    )

if __name__ == "__main__":
    app.run(debug=True)
