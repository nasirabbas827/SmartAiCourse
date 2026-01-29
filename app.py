from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
import sqlite3
import os
import json
from datetime import datetime

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
app.secret_key = 'smart-course-secure-key-2026' # Keep this secret for session encryption

# --- 1. DATA PREPARATION ---
# We load the dataset at startup so the app is fast during user requests
DATA_PATH = 'Courses_dataset.csv'

def load_and_clean_data():
    try:
        data = pd.read_csv(DATA_PATH)
        # Drop duplicates to keep recommendations unique
        data = data.drop_duplicates(subset=['course_title'])
        # Fill empty slots so the code doesn't crash on null values
        data['course_title'] = data['course_title'].fillna('Untitled Course')
        data['subject'] = data['subject'].fillna('General')
        # We combine title and subject to give the models more context for matching
        data['description_clean'] = data['course_title'].str.lower() + " " + data['subject'].str.lower()
        return data
    except Exception as e:
        print(f"Developer Note: Check if {DATA_PATH} exists! Error: {e}")
        # Return an empty dataframe structure so the app doesn't break entirely
        return pd.DataFrame(columns=['course_title', 'subject', 'description_clean', 'level'])

df = load_and_clean_data()

# --- 2. MODEL INITIALIZATION ---
# TF-IDF: Great for exact keyword matching (e.g., "Python", "React")
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=10000)
tfidf_matrix = tfidf.fit_transform(df['description_clean'])

# Neural Model: Great for understanding intent (e.g., "coding" matches "programming")
# Using 'all-MiniLM-L6-v2' as it's the perfect balance of speed and accuracy
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = semantic_model.encode(df['description_clean'].tolist(), convert_to_tensor=True)

# --- 3. DATABASE LOGIC ---
DB_PATH = 'smartcourse.db'

def get_db():
    """Connects to the SQLite database and allows accessing columns by name."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create tables if they don't exist yet."""
    with get_db() as conn:
        # Table for the search history log
        conn.execute('''CREATE TABLE IF NOT EXISTS searches
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, 
                      query TEXT, model TEXT, timestamp TEXT, results TEXT)''')
        # Table for the specific courses users want to keep
        conn.execute('''CREATE TABLE IF NOT EXISTS saved
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, 
                      query TEXT, course_id INTEGER, course_title TEXT, 
                      model TEXT, saved_at TEXT)''')

init_db()

@app.before_request
def ensure_session():
    """Ensures every visitor has a unique session ID for their history."""
    if 'session_id' not in session:
        session['session_id'] = os.urandom(16).hex()

# --- 4. WEB ROUTES ---

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

        # Logic for Keyword vs. Semantic matching
        if model_type == 'tfidf':
            query_vec = tfidf.transform([query.lower()])
            sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        else:
            query_emb = semantic_model.encode(query.lower(), convert_to_tensor=True)
            sim_scores = util.cos_sim(query_emb, embeddings)[0].cpu().numpy()

        # Sort and pick the top 10 most relevant courses
        top_indices = sim_scores.argsort()[-10:][::-1]
        results = []
        
        for idx in top_indices:
            score = round(float(sim_scores[idx]) * 100, 1)
            # Minimal score floor just for UI visual appeal
            if score < 1: score = 5.0 
            
            row = df.iloc[idx]
            results.append({
                'course_id': int(idx),
                'title': row['course_title'],
                'department': row['subject'],
                'description': (row['course_title'] + " " + row['subject'])[:150] + "...",
                'level': row.get('level', 'All Levels'),
                'score': score
            })

        # Save this search to the database history
        try:
            with get_db() as conn:
                conn.execute('INSERT INTO searches (session_id, query, model, timestamp, results) VALUES (?,?,?,?,?)',
                             (session['session_id'], query, model_type, 
                              datetime.now().strftime("%Y-%m-%d %H:%M"), 
                              json.dumps(results[:3]))) # Store top 3 as a preview string
                conn.commit()
        except Exception as e:
            print(f"History Log Error: {e}")

        return render_template('recommend.html', results=results, query=query, model_type=model_type)

    return render_template('recommend.html')

@app.route('/dashboard')
def dashboard():
    """Fetches user history and saved items for the dashboard view."""
    with get_db() as conn:
        history = conn.execute('SELECT * FROM searches WHERE session_id = ? ORDER BY id DESC', 
                               (session['session_id'],)).fetchall()
        saved = conn.execute('SELECT * FROM saved WHERE session_id = ? ORDER BY id DESC', 
                             (session['session_id'],)).fetchall()
    return render_template('dashboard.html', history=history, saved=saved)

@app.route('/clear_history')
def clear_history():
    """Wipes all search logs for the current user."""
    with get_db() as conn:
        conn.execute('DELETE FROM searches WHERE session_id = ?', (session['session_id'],))
        conn.commit()
    return redirect(url_for('dashboard'))

@app.route('/remove_saved/<int:item_id>')
def remove_saved(item_id):
    """Deletes a specific saved course."""
    with get_db() as conn:
        conn.execute('DELETE FROM saved WHERE id = ? AND session_id = ?', (item_id, session['session_id']))
        conn.commit()
    return redirect(url_for('dashboard'))

@app.route('/about')
def about():
    # Pass 'df' so we can show the total count of courses in the dataset
    return render_template('about.html', df=df)

# --- 5. API ENDPOINTS ---

@app.route('/api/save', methods=['POST'])
def api_save():
    """Endpoint called by the 'Save' button via JavaScript."""
    data = request.json
    try:
        with get_db() as conn:
            conn.execute('INSERT INTO saved (session_id, query, course_id, course_title, model, saved_at) VALUES (?,?,?,?,?,?)',
                         (session['session_id'], data['query'], data['course_id'], 
                          data['title'], data['model'], datetime.now().isoformat()))
            conn.commit()
        return jsonify({'status': 'saved'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print("--- SmartCourse System Starting Up ---")
    app.run(debug=True, port=5001)