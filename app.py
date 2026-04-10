import os
import pickle
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from urllib.parse import quote_plus

import requests
from flask import Flask, jsonify, redirect, render_template, request, session, url_for


BASE_DIR = Path(__file__).resolve().parent
TS_FMT = "%d %b %Y, %H:%M UTC"
HIGH_SIM = 0.35
MEDIUM_SIM = 0.18
LOW_EVIDENCE = 0.08
MODEL = None
VECTORIZER = None
MODEL_AVAILABLE = False

STOPWORDS = {
	"this", "that", "with", "from", "have", "has", "were", "was", "will", "would", "their",
	"about", "after", "before", "into", "over", "under", "between", "during", "there", "which",
	"when", "where", "your", "news", "claim", "said", "says", "report", "reported", "today"
}

MOCK_NEWS = [
	{"title": "RCB crowned as IPL 2025 champions", "summary": "RCB won the title after a strong campaign.", "source": "Mock News", "published": "" , "link": ""},
	{"title": "Government releases annual budget report", "summary": "An official report was published today.", "source": "Mock News", "published": "", "link": ""},
	{"title": "New policy announced by the ministry", "summary": "The ministry confirmed the new policy in a statement.", "source": "Mock News", "published": "", "link": ""},
]

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "vaartha-dev-key")


def clean(text):
	return re.sub(r"\s+", " ", (text or "").strip())


APP_LOGIN_PASSWORD = "vaartha123"


def now_utc():
	return datetime.now(timezone.utc)


def is_logged_in():
	return bool(session.get("user_email"))


def render_login(error="", next_path="/", login_name="", status_code=200):
	return render_template(
		"login.html",
		error=error,
		next_path=next_path,
		login_name=login_name,
	), status_code


def login_required(view):
	@wraps(view)
	def wrapped(*args, **kwargs):
		if is_logged_in():
			return view(*args, **kwargs)
		next_path = request.path if request.path != "/login" else "/"
		return redirect(url_for("login", next=next_path))
	return wrapped


def load_local_model():
	global MODEL, VECTORIZER, MODEL_AVAILABLE
	if MODEL is not None and VECTORIZER is not None:
		return True
	if MODEL_AVAILABLE is False and MODEL is None and VECTORIZER is None:
		try:
			with open(BASE_DIR / "model.pkl", "rb") as f:
				MODEL = pickle.load(f)
			with open(BASE_DIR / "vectorizer.pkl", "rb") as f:
				VECTORIZER = pickle.load(f)
			MODEL_AVAILABLE = True
		except Exception:
			MODEL_AVAILABLE = False
			MODEL = None
			VECTORIZER = None
	return MODEL_AVAILABLE


def tokens(text):
	return {t for t in re.findall(r"[a-z0-9]+", clean(text).lower()) if len(t) > 2}


def news_query(text):
	words = [w for w in re.findall(r"[a-z0-9]+", clean(text).lower()) if len(w) > 2 and w not in STOPWORDS]
	if not words:
		return "latest news"
	# Use a short keyword query so RSS returns relevant headlines more often.
	return " ".join(words[:8])





def fetch_news(query, limit=8):
	url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"
	try:
		r = requests.get(url, timeout=8)
		r.raise_for_status()
		root = ET.fromstring(r.text)
		items = []
		for item in root.findall(".//item")[:limit]:
			title = clean(item.findtext("title") or "")
			if not title:
				continue
			source_node = item.find("source")
			items.append({
				"title": title,
				"summary": clean(item.findtext("description") or ""),
				"source": clean(source_node.text if source_node is not None else ""),
				"published": clean(item.findtext("pubDate") or ""),
				"link": clean(item.findtext("link") or ""),
			})
		return items or MOCK_NEWS[:limit]
	except Exception:
		return MOCK_NEWS[:limit]


def headline_text(item):
	return clean(" ".join(filter(None, [item.get("title", ""), item.get("summary", ""), item.get("source", ""), item.get("published", "")])) )


def semantic_match(text, items):
	if not items:
		return None, 0.0
	try:
		query_tokens = tokens(text)
		if not query_tokens:
			return None, 0.0
		best_item = None
		best_score = 0.0
		for item in items:
			candidate = tokens(headline_text(item))
			if not candidate:
				continue
			intersection = len(query_tokens & candidate)
			union = len(query_tokens | candidate)
			jaccard = (intersection / union) if union else 0.0
			coverage = (intersection / min(len(query_tokens), len(candidate))) if min(len(query_tokens), len(candidate)) else 0.0
			# Blended score favors shared key terms while reducing very short accidental matches.
			score = (0.65 * coverage) + (0.35 * jaccard)
			if score > best_score:
				best_score = score
				best_item = item
		return best_item, float(best_score)
	except Exception:
		# Fallback if scoring fails
		return None, 0.0


def verdict_from_score(score):
	if score >= HIGH_SIM:
		return "Real"
	if score >= MEDIUM_SIM:
		return "Misleading"
	return "Fake"


def model_predict(text):
	if not load_local_model():
		return None
	try:
		vec = VECTORIZER.transform([clean(text)])
		pred = int(MODEL.predict(vec)[0])
		proba = None
		if hasattr(MODEL, "predict_proba"):
			proba = MODEL.predict_proba(vec)[0]
			# Label mapping from training script: 1=Fake, 0=Real
			fake_prob = float(proba[1])
			real_prob = float(proba[0])
			confidence = max(fake_prob, real_prob)
		else:
			fake_prob = 1.0 if pred == 1 else 0.0
			real_prob = 1.0 - fake_prob
			confidence = 0.5
		return {
			"prediction": "Fake" if pred == 1 else "Real",
			"confidence": confidence,
			"fake_prob": fake_prob,
			"real_prob": real_prob,
		}
	except Exception:
		return None


def analyze(text):
	query = news_query(text)
	latest = fetch_news(query, limit=3)  # Reduced to 3 for Render free tier memory
	best_match, similarity = semantic_match(text, latest)
	overlap_pct = similarity * 100
	clean_input = clean(text).lower()

	def has_exact_match():
		if not clean_input:
			return False
		for item in latest:
			candidate = clean(headline_text(item)).lower()
			if not candidate:
				continue
			if clean_input == candidate:
				return True
			if len(clean_input) >= 35 and clean_input in candidate:
				return True
			if len(candidate) >= 35 and candidate in clean_input:
				return True
		return False

	exact_match = has_exact_match()

	evidence = [f"Live overlap score: {overlap_pct:.1f}%"]

	# Strict rule requested: strong/exact live-news match => Real; otherwise Fake.
	if exact_match or similarity >= HIGH_SIM:
		prediction = "Real"
		confidence = round(max(overlap_pct, 85.0), 1)
		analysis_source = "overlap-rule"
		evidence.append("Submitted text closely matches live news content.")
	else:
		prediction = "Fake"
		confidence = round(max((1.0 - similarity) * 100, 75.0), 1)
		analysis_source = "overlap-rule"
		evidence.append("No strong claim match was found in live news, so this is treated as fake.")

	reason = ""
	if prediction == "Real":
		reason = "The submitted claim strongly matches current live-news content."
	else:
		reason = "The submitted claim does not match live-news evidence, so it is marked fake."
	return {
		"prediction": prediction,
		"prediction_class": prediction.lower(),
		"confidence": confidence,
		"confidence_band": "high" if confidence >= 70 else ("medium" if confidence >= 40 else "low"),
		"reasoning": reason,
		"analysis_source": analysis_source,
		"analysis_date": now_utc().strftime(TS_FMT),
		"evidence_snippets": evidence,
		"generated_facts": [f"Best matching headline: {best_match['title']}"] if best_match else [],
		"latest_news": latest,
		"live_news_query": query,
		"live_news_match": best_match,
	}


def input_text_from_request():
	text = clean(request.form.get("news", ""))
	upload = request.files.get("file")
	if upload and upload.filename and upload.filename.lower().endswith(".txt"):
		try:
			text = clean(f"{text}\n{upload.read().decode('utf-8', errors='ignore')}")
		except Exception:
			pass
	return text


@app.get("/")
@login_required
def home():
	return render_template("index.html")


@app.get("/about")
@login_required
def about():
	return render_template("about.html")


@app.get("/services")
@login_required
def services():
	return render_template("services.html")


@app.get("/contact")
@login_required
def contact():
	return render_template("contact.html")


@app.route("/login", methods=["GET", "POST"])
def login():
	next_path = clean(request.args.get("next", "")) or clean(request.form.get("next", "")) or "/"
	if request.method == "GET":
		if is_logged_in():
			return redirect(next_path)
		return render_login(next_path=next_path)

	name = clean(request.form.get("name", ""))
	password = request.form.get("password", "")
	if not name or password != APP_LOGIN_PASSWORD:
		return render_login(error="Invalid name or password.", next_path=next_path, login_name=name, status_code=401)

	session["user_email"] = name
	session["auth_provider"] = "shared-password"
	return redirect(next_path)


@app.get("/logout")
def logout():
	session.clear()
	return redirect(url_for("login"))


@app.post("/result")
@login_required
def result():
	text = input_text_from_request()
	if not text:
		return render_template(
			"result.html",
			prediction="Misleading",
			prediction_class="misleading",
			confidence=0,
			confidence_band="low",
			reasoning="No text provided.",
			analysis_source="input",
			analysis_date=now_utc().strftime(TS_FMT),
			source_excerpt="Please submit text or a .txt file.",
			evidence_snippets=[],
			generated_facts=[],
			latest_news=[],
			live_news_query="",
			live_news_match=None,
		)

	output = analyze(text)
	output["source_excerpt"] = text[:1600]
	return render_template("result.html", **output)


@app.post("/api/predict")
def api_predict():
	payload = request.get_json(silent=True) or {}
	text = clean(payload.get("text", ""))
	if not text:
		return jsonify({"ok": False, "error": "text is required"}), 400
	output = analyze(text)
	output["ok"] = True
	return jsonify(output)


@app.get("/api/latest-news")
def api_latest_news():
	query = clean(request.args.get("q", "latest headlines"))
	return jsonify({"ok": True, "query": query, "items": fetch_news(query)})


@app.get("/api/health")
@app.get("/health")
def health():
	return jsonify({
		"ok": True,
		"model_loaded": load_local_model(),
		"model_state": "local-ml" if MODEL_AVAILABLE else "lightweight-fallback",
		"time": now_utc().isoformat(),
	})


if __name__ == "__main__":
	port = int(os.getenv("PORT", "5000"))
	app.run(host="0.0.0.0", port=port, debug=False)
