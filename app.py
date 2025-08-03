from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
with open("spam_vML.pkl", "rb") as f:
    vectorizer = pickle.load(f)
    
with open("spam_mML.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])  # Handle both GET and POST
def home():
    prediction = None
    if request.method == 'POST':
        message = request.form.get('message', '').strip()
        if not message:
            return render_template('index.html', error="Please enter a message")
        
        try:
            features = vectorizer.transform([message])
            prediction = model.predict(features)[0]
        except Exception as e:
            return render_template('index.html', error=str(e))
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True,port=5000)
