from transformers import pipeline, set_seed
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask import Flask, request, jsonify
import torch

set_seed(42)  # Setting a random seed for reproducibility

def predictNewData(link):
    checkpoint = 'bgspaditya/malurl-electra-10e'
    id2label = {0:'benign',1:'defacement',2:'malware',3:'phishing'}
    label2id = {'benign':0,'defacement':1,'malware':2,'phishing':3}
    num_labels=4
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True, force_download=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels, id2label=id2label, label2id=label2id, force_download=True)
    url_classifier = pipeline(task='text-classification', model=model, tokenizer=tokenizer)
    result = url_classifier(link)
    return {'label': result[0]['label'], 'score': result[0]['score']}

app = Flask(__name__)

@app.route('/')
def api_home():
    return jsonify({"greeting": "Welcome to Malicious URL Detection using ELECTRA Api!"})

@app.route('/predict', methods=['POST'])
def predict_url():
    data = request.json
    text = data['text']
    prediction = predictNewData(text)
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)
