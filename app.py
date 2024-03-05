from flask import Flask, render_template, request
from textblob import TextBlob
from transformers import pipeline
from newspaper import Article

app = Flask(__name__)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

@app.route('/', methods=['GET', 'POST'])
def summarize():
    if request.method == 'POST':
        article_url = request.form['article_url']
        article_text = fetch_and_preprocess(article_url)
        original_text = article_text
        summary = generate_summary(article_text)
        sentiment = analyze_sentiment(article_text)
        return render_template('result.html',original_text=original_text,summary=summary, sentiment=sentiment)

    return render_template('index.html')

def fetch_and_preprocess(article_url):
    article = Article(article_url)
    article.download()
    article.parse()
    text = article.text
    return text

def generate_summary(article_text):
    max_input_length = 1024
    if len(article_text) > max_input_length:
        article_text = article_text[:max_input_length]

    summary = summarizer(article_text, max_length=500, min_length=200, length_penalty=2.0, num_beams=4)[0]['summary_text']
    return summary

if __name__ == '__main__':
    app.run(debug=True)
