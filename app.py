from sentence_transformers import SentenceTransformer
from tika import parser
from elasticsearch import Elasticsearch
import os
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, jsonify

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")
model = AutoModel.from_pretrained("vinai/phobert-large")

directory = './data'

app = Flask(__name__)
es = Elasticsearch(
    hosts=["https://localhost:9200"],  
    basic_auth=("elastic", "=T=ZB_DrZ3X*9cLRLBbT"),
    verify_certs=False  
)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

def remove_stopwords(query, stop_words):
    tokens = query.split() 
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return ' '.join(filtered_tokens)

def clean_text(query):
    query = query.lower()
    query = re.sub(r'[^a-zA-Z0-9\s]', '', query)
    return query

def lemmatize_text(query):
    tokens = query.split()  # Tách từ
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Gộp từ
    return ' '.join(lemmatized_tokens)
def preprocess_with_bert(query):
    cleaned_query = clean_text(query)
    lemmatized_query = lemmatize_text(cleaned_query)
    final_query = remove_stopwords(lemmatized_query, stop_words)
    input_ids = tokenizer.encode(final_query, return_tensors='pt', truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(input_ids)
    
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    
    return embeddings


def index_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            raw = parser.from_file(filepath)
            content = raw['content']
            
            if content:
                embedding = preprocess_with_bert(content)
                embedding = torch.tensor(embedding)
                embedding = torch.squeeze(embedding)
                doc = {
                    'filename': filename,
                    'content': content,
                    'embedding': embedding.tolist()  
                }
                es.index(index="files", body=doc)


def search_query(query):
    query_vector = preprocess_with_bert(query)
    query_vector = torch.tensor(query_vector)
    query_vector = torch.squeeze(query_vector)
    query_vector = query_vector.tolist()
    # print(query_vector)
    query = {
        "field": "embedding",
        "query_vector": query_vector,
        "k": 3, 
        "num_candidates": 10 
    }
    
    res = es.knn_search(index="files", knn=query)
    return res['hits']['hits']

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    query = data.get('query', '')
    
    if query:
        embeddings = preprocess_with_bert(query)
        embeddings = torch.tensor(embeddings)
        embeddings = torch.squeeze(embeddings)
        embeddings = embeddings.tolist()
        print("embeddings: ", embeddings)
        return jsonify({
            'embedded query': embeddings
        })
    else:
        return jsonify({'error': 'No query provided!'}), 400

@app.route('/extraction', methods=['POST'])
def file_extraction():
    index_files(directory)
    return jsonify({'message': 'Files indexed successfully'}), 200

@app.route('/search', methods=['POST'])
def searching():
    data = request.json
    query = data.get('query', '')
    results = search_query(query)
    output = []
    for result in results:
        output.append({
            "filename": result['_source']['filename'],
            "score": result['_score']
        })
    return jsonify(output)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
