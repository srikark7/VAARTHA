# VAARTHA

VAARTHA is a Flask news-classification app that labels a story as Real, Fake, or Unverified. It uses a local TF-IDF + Logistic Regression model, a built-in fact checker for known claims, and a live Google News RSS feed for current headline context.

## What Changed

- Added a live news verification layer backed by Google News RSS.
- Added a fact engine for common claims and sports records.
- Added a hard confidence gate so low-confidence model outputs become Unverified.
- Added working templates and styling for the full app.

## Features

- Paste news text or upload a `.txt` file.
- Get a verdict: Real, Fake, or Unverified.
- See live headlines used as current context during analysis.
- Fall back to the local model if no fact or live-news match is found.
- View a JSON API endpoint for integration tests.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
SECRET_KEY=your-flask-secret-key
PORT=5000
```

- No login environment variables are required.
- Users can log in with any name and password `vaartha123`.

### 3. Train or Refresh the Local Model

Run the training script if you want to rebuild `model.pkl` and `vectorizer.pkl`:

```bash
python model.py
```

### 4. Start the App

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## How the Verdict Works

1. The input text is cleaned and normalized.
2. The backend checks a built-in fact table for common claims.
3. The backend fetches recent Google News RSS headlines for live context.
4. If a live headline strongly matches the claim, the app can support a Real verdict.
5. Otherwise, the local classifier predicts Real or Fake, and low-confidence results are downgraded to Unverified.

## API Endpoints

- `GET /` - Home page
- `GET /about` - About page
- `GET /services` - Method overview
- `GET /contact` - Contact page
- `POST /result` - Analyze a news item
- `POST /api/predict` - JSON API for programmatic use
- `GET /api/latest-news` - Fetch live headlines from Google News RSS
- `GET /api/health` - Health check and model status

## Deploy On Render

1. Push this project to GitHub.
2. In Render, click New +, then Web Service.
3. Select your repository.
4. Render will detect `render.yaml` automatically.
5. Add these environment variables in Render Dashboard:

- `SECRET_KEY` (set a strong random value)

6. Click Deploy.
7. Open your Render URL and sign in.

For login:

- Use your name and password `vaartha123`.

## Project Files

- `app.py` - Flask backend and inference flow
- `model.py` - Training script for the local classifier
- `templates/` - HTML pages
- `static/` - Styles and front-end behavior
- `model.pkl` - Saved classifier
- `vectorizer.pkl` - Saved TF-IDF vectorizer
- `Fake.csv` / `True.csv` - Base datasets
- `Fake_updated.csv` / `True_updated.csv` - Updated datasets

## Notes

- Live news fetches use Google News RSS and do not require an API key.
- The app caches live headlines briefly to avoid repeated requests.
- Low-confidence outputs are treated as Unverified instead of forcing a Real/Fake label.

