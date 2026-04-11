import os
import pickle
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
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


def published_dt_utc(item):
	raw = clean((item or {}).get("published", ""))
	if not raw:
		return None
	try:
		dt = parsedate_to_datetime(raw)
		if dt is None:
			return None
		if dt.tzinfo is None:
			dt = dt.replace(tzinfo=timezone.utc)
		return dt.astimezone(timezone.utc)
	except Exception:
		return None


def recency_weight(item):
	dt = published_dt_utc(item)
	if dt is None:
		return 1.0
	age_days = max((now_utc() - dt).total_seconds() / 86400.0, 0.0)
	if age_days <= 2:
		return 1.10
	if age_days <= 7:
		return 1.00
	if age_days <= 30:
		return 0.92
	if age_days <= 90:
		return 0.80
	return 0.65


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


def score_tokens(query_tokens, candidate_tokens):
	intersection = len(query_tokens & candidate_tokens)
	union = len(query_tokens | candidate_tokens)
	jaccard = (intersection / union) if union else 0.0
	coverage = (intersection / min(len(query_tokens), len(candidate_tokens))) if min(len(query_tokens), len(candidate_tokens)) else 0.0
	return (0.65 * coverage) + (0.35 * jaccard), jaccard, coverage


def news_query(text):
	words = [w for w in re.findall(r"[a-z0-9]+", clean(text).lower()) if len(w) > 2 and w not in STOPWORDS]
	if not words:
		return "latest news"
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
	return clean(" ".join(filter(None, [item.get("title", ""), item.get("summary", "")])) )


def semantic_match(text, items):
	if not items:
		return None, 0.0
	query_tokens = tokens(text)
	if not query_tokens:
		return None, 0.0
	best_item, best_score = None, 0.0
	for item in items:
		candidate = tokens(headline_text(item))
		if not candidate:
			continue
		score, _, _ = score_tokens(query_tokens, candidate)
		if score > best_score:
			best_item, best_score = item, score
	return best_item, float(best_score)


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


def count_matching_sources(text, news_items):
	if not news_items:
		return [], [], [], None, 0.0

	source_scores, best_match, best_score = {}, None, 0.0
	query_tokens = tokens(text)
	if not query_tokens:
		return [], [], [], None, 0.0
	for item in news_items:
		candidate = tokens(headline_text(item))
		if not candidate:
			continue
		score, _, _ = score_tokens(query_tokens, candidate)
		score = min(1.0, score * recency_weight(item))
		if score > best_score:
			best_match, best_score = item, score
		source_scores.setdefault(item.get("source", "Unknown"), []).append((score, item))
	strong = {src: round(max(v[0] for v in vals) * 100, 1) for src, vals in source_scores.items() if max(v[0] for v in vals) >= 0.15}
	medium = {src: round(max(v[0] for v in vals) * 100, 1) for src, vals in source_scores.items() if 0.08 <= max(v[0] for v in vals) < 0.15}
	weak = {src: round(max(v[0] for v in vals) * 100, 1) for src, vals in source_scores.items() if max(v[0] for v in vals) < 0.08}
	return sorted(strong.items(), key=lambda x: x[1], reverse=True), sorted(medium.items(), key=lambda x: x[1], reverse=True), sorted(weak.items(), key=lambda x: x[1], reverse=True), best_match, best_score


def check_contradictions(items, text):
	if len(items) < 2:
		return False
	try:
		item_tokens = [tokens(headline_text(item)) for item in items]
		overlaps = [len(item_tokens[i] & item_tokens[j]) / len(item_tokens[i] | item_tokens[j]) for i in range(len(item_tokens)) for j in range(i + 1, len(item_tokens)) if item_tokens[i] and item_tokens[j] and len(item_tokens[i] | item_tokens[j]) > 0]
		return bool(overlaps and sum(overlaps) / len(overlaps) < 0.1)
	except Exception:
		return False


def debunking_sources(items):
	keywords = {
		"fact check", "fact-check", "fact checked", "debunk", "hoax", "false claim", "fake news",
		"misleading claim", "rumour", "rumor", "viral claim", "not true", "myth", "myths",
		"no evidence", "lacks evidence", "baseless", "conspiracy", "fact-checked",
		"doesn't", "does not", "don't", "not possible", "won't", "wrong"
	}
	sources = set()
	for item in items or []:
		blob = headline_text(item).lower()
		if any(k in blob for k in keywords):
			src = clean(item.get("source", "")) or "Unknown"
			sources.add(src)
	return sorted(sources)


def is_debunk_context_claim(text):
	blob = clean(text).lower()
	markers = (
		"fabricated", "debunked", "debunk", "hoax", "conspiracy", "false claim",
		"fake news", "baseless", "rumour", "rumor", "no evidence", "not true", "myth"
	)
	return any(marker in blob for marker in markers)


def is_death_claim(text):
	blob = clean(text).lower()
	death_terms = (" is dead", " died", " dies", " death", " passed away", " killed", " assassinated")
	return any(term in blob for term in death_terms)


def is_alive_claim(text):
	blob = clean(text).lower()
	alive_terms = (" is alive", " alive", " still alive", " living", " is living")
	return any(term in blob for term in alive_terms)


def death_claim_support(text, items):
	if not items:
		return 0, 0, []

	death_words = {"dead", "died", "dies", "death", "killed", "assassinated", "passed", "away"}
	neutral_words = {"pm", "cm", "mr", "mrs", "minister", "prime", "leader", "sir", "madam", "india", "indian"}
	claim_tokens = tokens(text)
	subject_tokens = claim_tokens - death_words - neutral_words
	if not subject_tokens:
		subject_tokens = claim_tokens - death_words

	debunk_markers = {"fact check", "fact-check", "hoax", "myth", "no evidence", "baseless", "conspiracy", "not true", "false claim", "does not", "doesn't", "don't", "fake"}

	sources = set()
	recent_sources = set()
	for item in items:
		blob = headline_text(item).lower()
		cand = tokens(blob)
		if not cand:
			continue
		if not (cand & death_words):
			continue
		if subject_tokens and not (cand & subject_tokens):
			continue
		if any(m in blob for m in debunk_markers):
			continue

		words = re.findall(r"[a-z0-9]+", blob)
		subject_positions = [i for i, w in enumerate(words) if w in subject_tokens]
		death_positions = [i for i, w in enumerate(words) if w in death_words]
		if subject_positions and death_positions:
			min_gap = min(abs(s - d) for s in subject_positions for d in death_positions)
			if min_gap > 3:
				continue

		source = clean(item.get("source", "")) or "Unknown"
		sources.add(source)
		dt = published_dt_utc(item)
		if dt is not None and (now_utc() - dt).total_seconds() <= (7 * 86400):
			recent_sources.add(source)

	return len(sources), len(recent_sources), sorted(sources)


def direct_claim_support(text, items):
	if not items:
		return 0, 0, []

	query_tokens = tokens(text)
	if not query_tokens:
		return 0, 0, []

	support_sources = set()
	recent_sources = set()
	for item in items:
		headline = headline_text(item)
		candidate = tokens(headline)
		if not candidate:
			continue

		intersection = len(query_tokens & candidate)
		union = len(query_tokens | candidate)
		jaccard = (intersection / union) if union else 0.0
		coverage = (intersection / min(len(query_tokens), len(candidate))) if min(len(query_tokens), len(candidate)) else 0.0

		if coverage >= 0.55 and jaccard >= 0.20:
			source = clean(item.get("source", "")) or "Unknown"
			support_sources.add(source)
			dt = published_dt_utc(item)
			if dt is not None and (now_utc() - dt).total_seconds() <= (14 * 86400):
				recent_sources.add(source)

	return len(support_sources), len(recent_sources), sorted(support_sources)


def broad_live_spread(text, items):
	if not items:
		return 0, 0, []

	query_tokens = tokens(text)
	if not query_tokens:
		return 0, 0, []

	sources = set()
	matching_items = 0
	for item in items:
		candidate = tokens(headline_text(item))
		if not candidate:
			continue
		score, _, _ = score_tokens(query_tokens, candidate)
		if score < 0.10:
			continue
		matching_items += 1
		sources.add(clean(item.get("source", "")) or "Unknown")
	return len(sources), matching_items, sorted(sources)


def exact_live_headline_match(text, items):
	if not items:
		return None, 0.0, []

	query_text = clean(text).lower()
	query_tokens = tokens(text)
	best_item = None
	best_score = 0.0
	matched_sources = []

	for item in items:
		headline = clean(item.get("title", "")).lower()
		if not headline:
			continue
		headline_tokens = tokens(headline)
		if not headline_tokens:
			continue
		score, jaccard, coverage = score_tokens(query_tokens, headline_tokens)
		normalized_headline = re.sub(r"[^a-z0-9]+", " ", headline).strip()
		normalized_query = re.sub(r"[^a-z0-9]+", " ", query_text).strip()
		exactish = normalized_query == normalized_headline or normalized_query in normalized_headline or normalized_headline in normalized_query
		if exactish or (coverage >= 0.80 and jaccard >= 0.40) or score >= 0.70:
			source = clean(item.get("source", "")) or "Unknown"
			if score > best_score:
				best_item = item
				best_score = score
			if source not in matched_sources:
				matched_sources.append(source)

	return best_item, best_score, matched_sources


def opposite_evidence(text, items):
	blob = clean(text).lower()
	if not items:
		return False
	for item in items:
		headline = headline_text(item).lower()
		if any(w in blob for w in ("low", "falls", "fall", "decrease", "down")) and any(w in headline for w in ("high", "rise", "rising", "record high", "increase", "up")):
			return True
		if any(w in blob for w in ("high", "rise", "rising", "increase", "up")) and any(w in headline for w in ("low", "fall", "falls", "decrease", "down")):
			return True
		if any(w in blob for w in ("dead", "death", "died", "killed")) and any(w in headline for w in ("alive", "living", "still alive")):
			return True
		if any(w in blob for w in ("alive", "living", "still alive")) and any(w in headline for w in ("dead", "death", "died", "killed")):
			return True
		if any(w in blob for w in ("lost", "loses")) and any(w in headline for w in ("won", "winner", "champions")):
			return True
		if any(w in blob for w in ("won", "winner", "champions")) and any(w in headline for w in ("lost", "loses")):
			return True
	return False


def alive_claim_support(text, items):
	if not items:
		return 0, 0, []

	claim_tokens = tokens(text)
	if not claim_tokens:
		return 0, 0, []

	subject_tokens = claim_tokens - {"alive", "living", "still"}
	sources = set()
	recent_sources = set()
	for item in items:
		headline = headline_text(item)
		candidate = tokens(headline)
		if not candidate or not (candidate & subject_tokens):
			continue
		source = clean(item.get("source", "")) or "Unknown"
		sources.add(source)
		dt = published_dt_utc(item)
		if dt is not None and (now_utc() - dt).total_seconds() <= (14 * 86400):
			recent_sources.add(source)
	return len(sources), len(recent_sources), sorted(sources)


def extract_claim_numbers(text):
	nums = set()
	for token in re.findall(r"\b\d+\b", clean(text)):
		try:
			value = int(token)
		except Exception:
			continue
		# Ignore likely years to reduce false mismatches (e.g., 2025/2026)
		if 1900 <= value <= 2100:
			continue
		nums.add(value)
	return nums


def numeric_claim_conflict(text, items):
	claim_numbers = extract_claim_numbers(text)
	if not claim_numbers or not items:
		return False, sorted(claim_numbers), []

	query_tokens = tokens(text)
	if not query_tokens:
		return False, sorted(claim_numbers), []

	live_numbers = set()
	for item in items:
		headline = headline_text(item)
		candidate = tokens(headline)
		if not candidate:
			continue
		intersection = len(query_tokens & candidate)
		union = len(query_tokens | candidate)
		jaccard = (intersection / union) if union else 0.0
		coverage = (intersection / min(len(query_tokens), len(candidate))) if min(len(query_tokens), len(candidate)) else 0.0
		relevance_score = (0.65 * coverage) + (0.35 * jaccard)
		if relevance_score < 0.12:
			continue
		live_numbers.update(extract_claim_numbers(headline))

	if not live_numbers:
		return False, sorted(claim_numbers), []

	has_conflict = len(claim_numbers & live_numbers) == 0
	return has_conflict, sorted(claim_numbers), sorted(live_numbers)


def build_result(
	prediction,
	confidence,
	analysis_source,
	evidence,
	best_match,
	latest,
	query,
	strong_sources,
	medium_sources,
	weak_sources,
	has_contradictions,
	numeric_conflict,
	claim_numbers,
	live_numbers,
	direct_support_count,
	direct_recent_count,
	direct_support_sources,
	alive_support_count,
	alive_recent_count,
	alive_support_sources,
	overlap_score,
	reason,
	confidence_band=None,
):
	if confidence_band is None:
		confidence_band = "high" if confidence >= 70 else ("medium" if confidence >= 40 else "low")
	return {
		"prediction": prediction,
		"prediction_class": prediction.lower(),
		"confidence": confidence,
		"confidence_band": confidence_band,
		"reasoning": reason,
		"analysis_source": analysis_source,
		"analysis_date": now_utc().strftime(TS_FMT),
		"evidence_snippets": evidence,
		"generated_facts": [f"Best matching headline: {best_match['title']}"] if best_match else [],
		"latest_news": latest,
		"live_news_query": query,
		"live_news_match": best_match,
		"strong_sources": strong_sources,
		"medium_sources": medium_sources,
		"weak_sources": weak_sources,
		"has_contradictions": has_contradictions,
		"numeric_conflict": numeric_conflict,
		"claim_numbers": claim_numbers,
		"live_numbers": live_numbers,
		"direct_support_count": direct_support_count,
		"direct_recent_count": direct_recent_count,
		"direct_support_sources": direct_support_sources,
		"alive_support_count": alive_support_count,
		"alive_recent_count": alive_recent_count,
		"alive_support_sources": alive_support_sources,
		"overlap_score": overlap_score,
	}


def analyze(text):
	query = news_query(text)
	latest = fetch_news(query, limit=12)
	debunk_context_claim = is_debunk_context_claim(text)
	exact_item, exact_score, exact_sources = exact_live_headline_match(text, latest)
	if exact_item is not None:
		if debunk_context_claim:
			evidence = [f"Live overlap score: {exact_score * 100:.1f}%", "Claim text contains debunking/fabrication context markers."]
			if exact_sources:
				evidence.append(f"Headline sources: {', '.join(exact_sources[:3])}")
			return build_result(
				"Fake",
				round(max(85.0, min(95.0, exact_score * 100)), 1),
				"exact-live-debunk-context",
				evidence,
				exact_item,
				latest,
				query,
				[],
				[],
				[],
				False,
				False,
				[],
				[],
				0,
				0,
				[],
				0,
				0,
				[],
				exact_score,
				"The pasted prompt references a fabricated/debunked claim context.",
			)
		evidence = [f"Live overlap score: {exact_score * 100:.1f}%", "Exact or near-exact live headline match found."]
		if exact_sources:
			evidence.append(f"Headline sources: {', '.join(exact_sources[:3])}")
		return build_result(
			"Real",
			round(max(exact_score * 100, 85.0), 1),
			"exact-live-headline",
			evidence,
			exact_item,
			latest,
			query,
			[],
			[],
			[],
			False,
			False,
			[],
			[],
			0,
			0,
			[],
			0,
			0,
			[],
			exact_score,
			"The pasted prompt matches a current Google News headline.",
		)
	alive_claim = is_alive_claim(text)
	death_claim = is_death_claim(text)
	if alive_claim:
		alive_support_count, alive_recent_count, alive_support_sources = alive_claim_support(text, latest)
		if alive_support_count >= 1 and not death_claim:
			best_item, best_score = semantic_match(text, latest)
			overlap_pct = best_score * 100
			evidence = [f"Live overlap score: {overlap_pct:.1f}%", f"Trusted live coverage mentions the subject in {alive_support_count} sources."]
			if alive_support_sources:
				evidence.append(f"Relevant sources: {', '.join(alive_support_sources[:3])}")
			return build_result(
				"Real",
				round(max(overlap_pct, 85.0), 1),
				"alive-current-support",
				evidence,
				best_item,
				latest,
				query,
				[],
				[],
				[],
				False,
				False,
				[],
				[],
				0,
				0,
				[],
				alive_support_count,
				alive_recent_count,
				alive_support_sources,
				best_score,
				"Current trusted coverage supports the person's active/alive status and no death evidence is present.",
			)

	strong_sources, medium_sources, weak_sources, best_match, best_score = count_matching_sources(text, latest)
	overlap_pct = best_score * 100

	strong_items = latest if strong_sources else []
	has_contradictions = check_contradictions(strong_items[:3], text) if strong_sources else False

	evidence = [f"Live overlap score: {overlap_pct:.1f}%"]

	trusted_from_medium = [(src, score) for src, score in medium_sources if score >= 10.0]
	trusted_support_count = len(strong_sources) + len(trusted_from_medium)
	debunk_sources = debunking_sources(latest)
	debunk_count = len(debunk_sources)
	death_claim = is_death_claim(text)
	alive_claim = is_alive_claim(text)
	death_support_count, death_recent_count, death_support_sources = death_claim_support(text, latest)
	alive_support_count, alive_recent_count, alive_support_sources = alive_claim_support(text, latest)
	direct_support_count, direct_recent_count, direct_support_sources = direct_claim_support(text, latest)
	broad_source_count, broad_item_count, broad_sources = broad_live_spread(text, latest)
	has_opposite_evidence = opposite_evidence(text, latest)
	numeric_conflict, claim_numbers, live_numbers = numeric_claim_conflict(text, latest)
	trusted_source_names = {src for src, _ in strong_sources} | {src for src, _ in trusted_from_medium}
	recent_trusted_sources = set()
	for item in latest:
		source = clean(item.get("source", "")) or "Unknown"
		if source not in trusted_source_names:
			continue
		dt = published_dt_utc(item)
		if dt is not None and (now_utc() - dt).total_seconds() <= (14 * 86400):
			recent_trusted_sources.add(source)
	recent_trusted_count = len(recent_trusted_sources)
	very_low_confidence = best_score < 0.10
	semantic_mismatch = best_score >= 0.20 and direct_support_count == 0

	# Classification logic: REAL / FAKE / MISLEADING only
	if death_claim and debunk_count >= 1 and trusted_support_count >= 1:
		evidence.append(f"Death claim has debunk/fact-check coverage from {debunk_count} source(s).")
		return build_result("Fake", 90.0, "death-claim-debunked", evidence, best_match, latest, query, strong_sources, medium_sources, weak_sources, has_contradictions, numeric_conflict, claim_numbers, live_numbers, direct_support_count, direct_recent_count, direct_support_sources, alive_support_count, alive_recent_count, alive_support_sources, best_score, "The claim says a person is dead, but live coverage includes debunking signals.")

	elif debunk_context_claim and (direct_support_count == 0 or best_score < 0.18):
		evidence.append("Claim wording indicates fabricated/debunked conspiracy context.")
		if trusted_support_count > 0:
			evidence.append("Related coverage exists, but it does not directly verify the claim as true.")
		return build_result("Fake", round(max(82.0, min(95.0, overlap_pct)), 1), "debunk-context-weak-support", evidence, best_match, latest, query, strong_sources, medium_sources, weak_sources, has_contradictions, numeric_conflict, claim_numbers, live_numbers, direct_support_count, direct_recent_count, direct_support_sources, alive_support_count, alive_recent_count, alive_support_sources, best_score, "The prompt reflects a debunked/fabricated claim context and lacks direct trustworthy confirmation.")

	elif death_claim and (death_support_count < 2 or death_recent_count < 1):
		evidence.append(f"Direct death confirmation is insufficient (sources={death_support_count}, recent={death_recent_count}).")
		if death_support_sources:
			evidence.append(f"Direct supporting sources found: {', '.join(death_support_sources[:3])}")
		return build_result("Fake", 88.0, "death-claim-unconfirmed", evidence, best_match, latest, query, strong_sources, medium_sources, weak_sources, has_contradictions, numeric_conflict, claim_numbers, live_numbers, direct_support_count, direct_recent_count, direct_support_sources, alive_support_count, alive_recent_count, alive_support_sources, best_score, "A death claim requires direct, recent, multi-source confirmation, which is missing.")

	elif broad_source_count >= 2 and broad_item_count >= 2 and not has_contradictions and best_score >= 0.16 and direct_support_count >= 1:
		evidence.append(f"Live coverage is spread across {broad_source_count} sources and {broad_item_count} matching items.")
		if broad_sources:
			evidence.append(f"Broad sources: {', '.join(broad_sources[:3])}")
		return build_result("Real", round(max(overlap_pct, 80.0), 1), "broad-live-spread", evidence, best_match, latest, query, strong_sources, medium_sources, weak_sources, has_contradictions, numeric_conflict, claim_numbers, live_numbers, direct_support_count, direct_recent_count, direct_support_sources, alive_support_count, alive_recent_count, alive_support_sources, best_score, "The claim is widely covered across live news with sufficient direct semantic support.")

	elif numeric_conflict and trusted_support_count >= 1 and best_score >= 0.10:
		evidence.append(f"Numeric mismatch found. Claim numbers: {claim_numbers}; live evidence numbers: {live_numbers}.")
		return build_result("Fake", round(max(80.0, min(95.0, overlap_pct)), 1), "numeric-counterevidence", evidence, best_match, latest, query, strong_sources, medium_sources, weak_sources, has_contradictions, numeric_conflict, claim_numbers, live_numbers, direct_support_count, direct_recent_count, direct_support_sources, alive_support_count, alive_recent_count, alive_support_sources, best_score, "The topic appears in live news, but key numeric facts in the claim do not match live coverage.")

	elif alive_claim and alive_support_count >= 1 and trusted_support_count >= 1 and not death_claim and death_support_count == 0:
		evidence.append(f"Trusted live coverage mentions the subject in {alive_support_count} sources.")
		if alive_support_sources:
			evidence.append(f"Relevant sources: {', '.join(alive_support_sources[:3])}")
		return build_result("Real", round(max(overlap_pct, 85.0), 1), "alive-current-support", evidence, best_match, latest, query, strong_sources, medium_sources, weak_sources, has_contradictions, numeric_conflict, claim_numbers, live_numbers, direct_support_count, direct_recent_count, direct_support_sources, alive_support_count, alive_recent_count, alive_support_sources, best_score, "Current trusted coverage supports the person's active/alive status and no death evidence is present.")

	elif trusted_support_count >= 1 and direct_support_count == 0 and best_score > 0.36:
		evidence.append(f"Live overlap score is {overlap_pct:.1f}%, above the 36% real-support threshold.")
		evidence.append("Similarity is high, but the exact claim wording is only supported indirectly by trusted sources.")
		return build_result("Real", round(max(overlap_pct, 80.0), 1), "high-overlap-real", evidence, best_match, latest, query, strong_sources, medium_sources, weak_sources, has_contradictions, numeric_conflict, claim_numbers, live_numbers, direct_support_count, direct_recent_count, direct_support_sources, alive_support_count, alive_recent_count, alive_support_sources, best_score, "The claim has strong live overlap and related trusted coverage even though the exact wording is not directly confirmed.")

	elif semantic_mismatch:
		evidence.append("Similarity is high but exact claim meaning is not confirmed by trusted sources.")
		return build_result("Fake", round(max(80.0, min(95.0, overlap_pct)), 1), "semantic-mismatch", evidence, best_match, latest, query, strong_sources, medium_sources, weak_sources, has_contradictions, numeric_conflict, claim_numbers, live_numbers, direct_support_count, direct_recent_count, direct_support_sources, alive_support_count, alive_recent_count, alive_support_sources, best_score, "Related news exists, but trusted sources do not confirm the exact claim.")

	elif trusted_support_count == 0:
		evidence.append("No trusted source confirms this claim.")
		return build_result("Fake", round(max((1.0 - best_score) * 100, 78.0), 1), "no-trusted-support", evidence, best_match, latest, query, strong_sources, medium_sources, weak_sources, has_contradictions, numeric_conflict, claim_numbers, live_numbers, direct_support_count, direct_recent_count, direct_support_sources, alive_support_count, alive_recent_count, alive_support_sources, best_score, "The claim is not confirmed by trusted sources.")

	elif debunk_count >= 2 and trusted_support_count >= 1 and best_score >= 0.10:
		evidence.append(f"Counter-evidence found from {debunk_count} fact-check/debunk source(s).")
		evidence.append(f"Debunking sources: {', '.join(debunk_sources[:3])}")
		return build_result("Fake", round(max(80.0, min(95.0, overlap_pct)), 1), "fact-check-counterevidence", evidence, best_match, latest, query, strong_sources, medium_sources, weak_sources, has_contradictions, numeric_conflict, claim_numbers, live_numbers, direct_support_count, direct_recent_count, direct_support_sources, alive_support_count, alive_recent_count, alive_support_sources, best_score, "The claim appears in coverage, but multiple sources frame it as false or misleading.")

	elif trusted_support_count >= 3 and recent_trusted_count >= 1 and direct_support_count >= 1 and best_score >= 0.15 and not has_contradictions:
		evidence.append(f"Verified across {trusted_support_count} trusted news sources.")
		evidence.append(f"Recent trusted support (last 14 days): {recent_trusted_count} source(s).")
		evidence.append(f"Direct claim confirmations: {direct_support_count} source(s).")
		if strong_sources:
			evidence.append(f"Strong matches: {', '.join([f'{src} ({score}%)' for src, score in strong_sources[:3]])}")
		elif trusted_from_medium:
			evidence.append(f"Trusted medium matches: {', '.join([f'{src} ({score}%)' for src, score in trusted_from_medium[:3]])}")
		return build_result("Real", round(max(overlap_pct, 85.0), 1), "multi-source-verification", evidence, best_match, latest, query, strong_sources, medium_sources, weak_sources, has_contradictions, numeric_conflict, claim_numbers, live_numbers, direct_support_count, direct_recent_count, direct_support_sources, alive_support_count, alive_recent_count, alive_support_sources, best_score, f"The submitted claim is supported by {trusted_support_count} trusted news sources with consistent meaning.")

	elif has_contradictions or very_low_confidence:
		if has_contradictions:
			evidence.append("Conflicting reports found across trusted sources.")
		if very_low_confidence:
			evidence.append("Very low confidence signal for the submitted claim.")
		return build_result("Misleading", round(max(best_score * 100, 10.0), 1), "conflict-or-low-confidence", evidence, best_match, latest, query, strong_sources, medium_sources, weak_sources, has_contradictions, numeric_conflict, claim_numbers, live_numbers, direct_support_count, direct_recent_count, direct_support_sources, alive_support_count, alive_recent_count, alive_support_sources, best_score, "The claim has conflicting or weak support and may be missing context.")

	elif best_score < 0.08 or (trusted_support_count == 0 and not death_claim):
		if best_score < 0.08:
			evidence.append("Similarity is very low (< 0.08), indicating no meaningful live-news match.")
		if trusted_support_count == 0:
			evidence.append("No trusted source confirms this claim at the current time.")
		if weak_sources:
			evidence.append(f"Weak matches only: {', '.join([f'{src} ({score}%)' for src, score in weak_sources[:2]])}")
		return build_result("Fake", round(max((1.0 - best_score) * 100, 75.0), 1), "no-source-match", evidence, best_match, latest, query, strong_sources, medium_sources, weak_sources, has_contradictions, numeric_conflict, claim_numbers, live_numbers, direct_support_count, direct_recent_count, direct_support_sources, alive_support_count, alive_recent_count, alive_support_sources, best_score, "The submitted claim has no credible support and contradicts or lacks evidence in news sources.")

	elif (1 <= trusted_support_count <= 2) and (0.10 <= best_score <= 0.18) and not has_contradictions:
		evidence.append(f"Partially supported by {trusted_support_count} trusted source(s).")
		evidence.append("Similarity is moderate (0.10 to 0.18) and context appears incomplete/exaggerated.")
		return build_result("Misleading", round(best_score * 100, 1), "partial-match", evidence, best_match, latest, query, strong_sources, medium_sources, weak_sources, has_contradictions, numeric_conflict, claim_numbers, live_numbers, direct_support_count, direct_recent_count, direct_support_sources, alive_support_count, alive_recent_count, alive_support_sources, best_score, "The submitted claim is partially supported but lacks sufficient corroboration. Information may be incomplete, exaggerated, or missing context.")

	else:
		if has_contradictions:
			evidence.append("Conflicting coverage found across sources; information may be incomplete or inconsistent.")
		else:
			evidence.append("Partial or unclear support found; claim is not fully verified.")
		return build_result("Misleading", round(max(best_score * 100, 10.0), 1), "edge-case-misleading", evidence, best_match, latest, query, strong_sources, medium_sources, weak_sources, has_contradictions, numeric_conflict, claim_numbers, live_numbers, direct_support_count, direct_recent_count, direct_support_sources, alive_support_count, alive_recent_count, alive_support_sources, best_score, "The claim is partially supported but not strongly verified enough to be treated as real.")


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
