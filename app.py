import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus

import requests
from flask import Flask, jsonify, render_template, request
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
TS_FMT = "%d %b %Y, %H:%M UTC"
HIGH_SIM = 0.61
MEDIUM_SIM = 0.45
EMBEDDER = None

MOCK_NEWS = [
	{"title": "RCB crowned as IPL 2025 champions", "summary": "RCB won the title after a strong campaign.", "source": "Mock News", "published": "" , "link": ""},
	{"title": "Government releases annual budget report", "summary": "An official report was published today.", "source": "Mock News", "published": "", "link": ""},
	{"title": "New policy announced by the ministry", "summary": "The ministry confirmed the new policy in a statement.", "source": "Mock News", "published": "", "link": ""},
]

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "vaartha-dev-key")


def now_utc():
	return datetime.now(timezone.utc)


def clean(text):
	return re.sub(r"\s+", " ", (text or "").strip())


def get_embedder():
	global EMBEDDER
	if EMBEDDER is None:
		try:
			EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
		except Exception:
			EMBEDDER = False
	return None if EMBEDDER is False else EMBEDDER


def news_query(text):
	return clean(text)[:140] or "latest news"





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
	embedder = get_embedder()
	if not embedder or not items:
		return None, 0.0
	corpus = [clean(text)] + [headline_text(item) for item in items]
	embeddings = embedder.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
	scores = cosine_similarity(embeddings[:1], embeddings[1:])[0]
	best_index = int(scores.argmax()) if len(scores) else -1
	return (items[best_index], float(scores[best_index])) if best_index >= 0 else (None, 0.0)


def verdict_from_score(score):
	if score >= HIGH_SIM:
		return "Real"
	if score >= MEDIUM_SIM:
		return "Misleading"
	return "Fake"


def analyze(text):
	query = news_query(text)
	latest = fetch_news(query)
	best_match, similarity = semantic_match(text, latest)
	prediction = verdict_from_score(similarity)
	confidence = round(similarity * 100, 1)
	reason = ""
	if prediction == "Real":
		reason = "Semantic embedding analysis shows high similarity to live coverage—the claim and news headline likely convey the same meaning."
	elif prediction == "Misleading":
		reason = "Semantic similarity is moderate—the input relates to live news but differs enough in meaning to warrant caution."
	else:
		reason = "Semantic embedding analysis found low similarity to any live coverage—insufficient support for the claim."
	if not get_embedder():
		reason = "SentenceTransformer embedding model could not be loaded, analysis unavailable."
	return {
		"prediction": prediction,
		"prediction_class": prediction.lower(),
		"confidence": confidence,
		"confidence_band": "high" if similarity >= HIGH_SIM else ("medium" if similarity >= MEDIUM_SIM else "low"),
		"reasoning": reason,
		"analysis_source": "semantic",
		"analysis_date": now_utc().strftime(TS_FMT),
		"evidence_snippets": [f"Best semantic similarity: {similarity * 100:.1f}%"],
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
def home():
	return render_template("index.html")


@app.get("/about")
def about():
	return render_template("about.html")


@app.get("/services")
def services():
	return render_template("services.html")


@app.get("/contact")
def contact():
	return render_template("contact.html")


@app.post("/result")
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
	model_state = "ready" if EMBEDDER not in (None, False) else ("not-loaded" if EMBEDDER is None else "unavailable")
	return jsonify({
		"ok": True,
		"model_loaded": EMBEDDER not in (None, False),
		"model_state": model_state,
		"time": now_utc().isoformat(),
	})


if __name__ == "__main__":
	port = int(os.getenv("PORT", "5000"))
	app.run(host="0.0.0.0", port=port, debug=False)
