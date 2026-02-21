from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
import sqlite3
import os
import json
from datetime import datetime
import joblib
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)
app.secret_key = 'smart-course-secure-key-2026'

# --- 1. DATA PREPARATION ---
DATA_PATH = 'Courses_dataset.csv'

def load_and_clean_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates(subset=['course_title'])
    df['course_title'] = df['course_title'].fillna('Untitled Course')
    df['subject'] = df['subject'].fillna('General')
    df['level'] = df['level'].fillna('All Levels')
    df = df.reset_index(drop=True)
    df['text_content'] = df['course_title'].astype(str) + " " + df['subject'].astype(str) + " " + df['level'].astype(str)
    return df

df = load_and_clean_data()

# --- 2. TEXT PREPROCESSING (NLTK) ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok.isalpha() and tok not in stop_words]
    return " ".join(tokens)

df['processed_text'] = df['text_content'].apply(clean_text)

# --- 3. LOAD MODELS AND EMBEDDINGS ---
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
tfidf_matrix_train = sparse.load_npz("tfidf_matrix.npz")  # Fixed load
train_df = joblib.load("train_dataframe.pkl")
embedding_model = SentenceTransformer("sentence_transformer_model")
train_embeddings = np.load("neural_embeddings.npy")

# --- 4. DATABASE LOGIC ---
DB_PATH = 'smartcourse.db'

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS searches
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, 
                      query TEXT, model TEXT, timestamp TEXT, results TEXT)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS saved
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, 
                      query TEXT, course_id INTEGER, course_title TEXT, 
                      model TEXT, saved_at TEXT)''')

init_db()

@app.before_request
def ensure_session():
    if 'session_id' not in session:
        session['session_id'] = os.urandom(16).hex()

# --- 5. RECOMMENDATION FUNCTIONS ---
def recommend_tfidf(query, top_k=10):
    query_vec = tfidf_vectorizer.transform([clean_text(query)])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix_train).flatten()
    top_indices = sim_scores.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        row = train_df.iloc[idx]
        score = round(float(sim_scores[idx]) * 100, 1)
        if score < 1: score = 5.0
        results.append({
            'course_id': int(idx),
            'title': row['course_title'],
            'department': row['subject'],
            'description': (row['course_title'] + " " + row['subject'])[:150] + "...",
            'level': row.get('level', 'All Levels'),
            'score': score
        })
    return results

def recommend_neural(query, top_k=10):
    query_emb = embedding_model.encode([clean_text(query)])
    sim_scores = cosine_similarity(query_emb, train_embeddings).flatten()
    top_indices = sim_scores.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        row = train_df.iloc[idx]
        score = round(float(sim_scores[idx]) * 100, 1)
        if score < 1: score = 5.0
        results.append({
            'course_id': int(idx),
            'title': row['course_title'],
            'department': row['subject'],
            'description': (row['course_title'] + " " + row['subject'])[:150] + "...",
            'level': row.get('level', 'All Levels'),
            'score': score
        })
    return results

# --- 6. WEB ROUTES ---
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        model_type = request.form.get('model', 'tfidf')

        if not query:
            return render_template('recommend.html', error="Your learning goal cannot be empty.")

        if model_type == 'tfidf':
            results = recommend_tfidf(query)
        else:
            results = recommend_neural(query)

        # Save search history
        try:
            with get_db() as conn:
                conn.execute(
                    'INSERT INTO searches (session_id, query, model, timestamp, results) VALUES (?,?,?,?,?)',
                    (session['session_id'], query, model_type, datetime.now().strftime("%Y-%m-%d %H:%M"), 
                     json.dumps(results[:3]))
                )
                conn.commit()
        except Exception as e:
            print(f"History Log Error: {e}")

        return render_template('recommend.html', results=results, query=query, model_type=model_type)

    return render_template('recommend.html')

@app.route('/dashboard')
def dashboard():
    with get_db() as conn:
        history = conn.execute('SELECT * FROM searches WHERE session_id = ? ORDER BY id DESC', 
                               (session['session_id'],)).fetchall()
        saved = conn.execute('SELECT * FROM saved WHERE session_id = ? ORDER BY id DESC', 
                             (session['session_id'],)).fetchall()
    return render_template('dashboard.html', history=history, saved=saved)

@app.route('/clear_history')
def clear_history():
    with get_db() as conn:
        conn.execute('DELETE FROM searches WHERE session_id = ?', (session['session_id'],))
        conn.commit()
    return redirect(url_for('dashboard'))

@app.route('/remove_saved/<int:item_id>')
def remove_saved(item_id):
    with get_db() as conn:
        conn.execute('DELETE FROM saved WHERE id = ? AND session_id = ?', (item_id, session['session_id']))
        conn.commit()
    return redirect(url_for('dashboard'))

@app.route('/about')
def about():
    return render_template('about.html', df=df)

# --- 7. API ENDPOINTS ---
@app.route('/api/save', methods=['POST'])
def api_save():
    data = request.json
    try:
        with get_db() as conn:
            conn.execute(
                'INSERT INTO saved (session_id, query, course_id, course_title, model, saved_at) VALUES (?,?,?,?,?,?)',
                (session['session_id'], data['query'], data['course_id'], 
                 data['title'], data['model'], datetime.now().isoformat())
            )
            conn.commit()
        return jsonify({'status': 'saved'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print("--- SmartCourse System Starting Up ---")
    app.run(debug=True, port=5001)