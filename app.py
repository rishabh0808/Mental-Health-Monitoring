from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Initialize the AI Sentiment Analysis Pipeline
try:
    # Uses a lightweight, accurate BERT variant fine-tuned for sentiment analysis
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
except Exception as e:
    print(f"Error loading AI model: {e}")
    classifier = None

# Critical trigger words for immediate risk intervention
CRISIS_KEYWORDS = ["suicide", "kill myself", "end my life", "hopeless", "depressed", "self-harm", "hurt myself"]

def analyze_mental_health(text):
    text_lower = text.lower()
    
    # 1. Immediate Crisis Detection
    if any(keyword in text_lower for keyword in CRISIS_KEYWORDS):
        return {
            "status": "CRISIS_ALERT",
            "message": "It looks like you are going through a very tough time. Please know you are not alone. Please reach out for immediate, confidential support: Contact a suicide and crisis hotline or emergency services.",
            "score": 100.0
        }
    
    # 2. AI-driven Sentiment Engine
    if classifier:
        result = classifier(text)
        label = result[0]['label']
        score = round(result[0]['score'] * 100, 2)
        
        if label == "NEGATIVE":
            return {
                "status": "STRESSED_SAD",
                "message": f"Your logs indicate signs of stress or low mood (Confidence: {score}%). Consider practicing mindfulness, taking a break, or reaching out to someone you trust.",
                "score": score
            }
        else:
            return {
                "status": "POSITIVE_STABLE",
                "message": f"Your text reflects a positive or stable mindset (Confidence: {score}%). Keep maintaining these healthy habits!",
                "score": score
            }
    else:
        return {"status": "ERROR", "message": "The AI pipeline is currently unavailable.", "score": 0.0}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    user_text = data.get('text', '')
    
    if not user_text.strip():
        return jsonify({"status": "EMPTY", "message": "Input text cannot be blank."})
        
    analysis_result = analyze_mental_health(user_text)
    return jsonify(analysis_result)

if __name__ == '__main__':
    app.run(debug=True)

