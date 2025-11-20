import pickle
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
model = pickle.load(open('pac.pkl', 'rb'))
tfidf = TfidfVectorizer(stop_words='english')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    data = [news]

    vect = tfidf.fit_transform(data)
    credible_words = [
        "government", "official", "research", "report", "study",
        "minister", "university", "scientists", "press conference",
        "election commission", "department", "international agency",
        "world health organization", "WHO", "UN"
    ]
    
    adjusted_confidence = 0

    for word in credible_words:
        if word.lower() in news.lower():
            adjusted_confidence += 0.04 

    try:
        prediction = model.predict(vect)[0]
    except:
        prediction = "FAKE"

    if adjusted_confidence > 0.5:
        prediction = "REAL"
        
    prediction = str(prediction).upper()

    return render_template('prediction.html', prediction=prediction)



if __name__ == '__main__':
    app.run(debug=True)
