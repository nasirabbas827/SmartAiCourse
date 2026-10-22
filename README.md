# Smart AI Course Recommendation

A Flask course recommendation application using course data, text processing and similarity-based recommendation components.

## Features

- Course recommendations through a web form and API
- Recommendation history
- Saved courses and dashboard
- SQLite storage

## Technology

Python and Flask.

## Local setup

Clone the repository:

```bash
git clone https://github.com/nasirabbas827/SmartAiCourse.git
cd SmartAiCourse
```

Use a virtual environment:

```bash
python -m venv .venv
# Windows:
# .venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

Complete the project-specific setup below, then run from the repository root:

```bash
python app.py
```

Open `http://127.0.0.1:5001/`.

## Project-specific setup

Run on port **5001**. `app.py` also imports NLTK, so install it with `python -m pip install nltk`. It references `tfidf_vectorizer.pkl`, `tfidf_matrix.npz` and `train_dataframe.pkl`, which are absent from the checked root listing. Review `SmartCourse_Model-firstversion.ipynb` and export the expected artifacts before using the matching recommendation path. Sentence-transformer models and NLTK resources may require an initial download.

## Repository layout

| Path | Purpose |
|---|---|
| `Courses_dataset.csv` | Dataset |
| `SmartCourse_Model-firstversion.ipynb` | Notebook for experiments or model preparation |
| `app.py` | Flask application |
| `requirements.txt` | Python dependencies |
| `smartcourse.db` | Application database |
| `templates/` | HTML templates |

## Development notes

Setup steps describe local development. Configure your own database, mail and API credentials where applicable. Review debug settings and replace development secrets before deploying. The application was not executed as part of this documentation update.

Maintained by [Nasir Abbas](https://github.com/nasirabbas827).
