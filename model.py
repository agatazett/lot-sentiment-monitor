"""
model.py – Moduł NLP: klasyfikacja sentymentu i Crisis Score
=============================================================
Używa reguł słownikowych jako szybkiego baseline'u.
W sekcji TRYB PRODUKCYJNY znajdziesz instrukcję podłączenia
prawdziwego modelu transformerowego (HerBERT / XLM-RoBERTa).
"""

import re
import math
from datetime import datetime


# ─── SŁOWNIKI POLSKIE ─────────────────────────────────────────────────────────

NEG_WORDS = [
    "skandal", "katastrofa", "dramat", "hańba", "fatalne", "okropne",
    "straszne", "beznadziejne", "fatalny", "okropny", "straszny",
    "nigdy więcej", "zero", "brak", "nikt", "ignorowanie",
    "opóźniony", "opóźnienie", "odwołali", "odwołany", "odwołanie",
    "zagubiony", "zaginął", "zaginęła", "uszkodzony",
    "nieuprzejmy", "nieuprzejma", "chamski", "chamska",
    "reklamacja", "odszkodowanie", "żądam", "żąda", "prawnicy",
    "zawód", "rozczarowanie", "niekompetentny", "niekompetentna",
    "wstyd", "do niczego", "nie działa", "awaria", "chaos",
    "zostawili", "porzucili", "bez informacji", "dezinformacja",
]

POS_WORDS = [
    "świetny", "doskonały", "polecam", "fantastyczny", "super",
    "wspaniały", "perfekcyjny", "zadowolony", "zadowolona",
    "dobry", "miły", "miła", "profesjonalny", "profesjonalna",
    "punktualny", "punktualna", "wygodny", "wygodna",
    "smaczny", "smaczna", "dziękuję", "dziękuje", "poprawia",
    "najlepszy", "najlepsza", "rewelacyjny", "rewelacyjna",
    "komfortowy", "komfortowa", "sprawny", "sprawna", "polecam",
    "gorąco polecam", "bardzo zadowolony", "mile zaskoczony",
]

NEG_INTENSIFIERS = [
    "absolutny", "kompletny", "totalny", "całkowity", "zupełny",
    "!!!", "!!", "wstyd", "hańba", "skandal",
]

# ─── KATEGORIE ────────────────────────────────────────────────────────────────

CATEGORY_KEYWORDS = {
    "opóźnienie":    ["opóźni", "spóźni", "czeka", "godzin wcześ", "delay", "odwołan"],
    "bagaż":         ["bagaż", "walizk", "zagubi", "zaginął", "zaginęła", "uszkodzon", "pir"],
    "obsługa":       ["obsług", "stewardes", "personel", "załog", "nieuprzejm", "chamsk", "pracown"],
    "komunikacja":   ["komunikacj", "informacj", "nikt nic", "brak kontakt", "cisza", "nie odpowiad",
                      "bez wieści", "ignorow", "milczen"],
    "odszkodowanie": ["odszkodow", "ec261", "ec 261", "refund", "zwrot", "reklamacj",
                      "zwrot pieniędzy", "prawni"],
    "inne":          [],
}

HIGH_RISK_CATEGORIES = {"odszkodowanie", "komunikacja", "bagaż"}

# ─── MODEL SENTYMENTU ─────────────────────────────────────────────────────────

class SentimentModel:
    """
    Baseline model oparty na regułach słownikowych.

    ── TRYB PRODUKCYJNY ──────────────────────────────────────────────────────
    Aby użyć prawdziwego modelu HerBERT / XLM-RoBERTa, zainstaluj:

        pip install transformers torch sentencepiece

    Następnie odkomentuj poniższy blok w metodzie __init__:

        from transformers import pipeline
        self.pipe = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            # Alternatywnie dla polskiego:
            # model="allegro/herbert-base-cased"  ← wymaga fine-tuningu
        )

    I zastąp metodę _score_text() wywołaniem:

        result = self.pipe(text[:512])[0]
        label_map = {"LABEL_0": "negatywny",
                     "LABEL_1": "neutralny",
                     "LABEL_2": "pozytywny",
                     "negative": "negatywny",
                     "neutral": "neutralny",
                     "positive": "pozytywny"}
        return label_map.get(result["label"], "neutralny"), round(result["score"]*100)
    ──────────────────────────────────────────────────────────────────────────
    """

    def predict(self, df):
        """Przewiduje sentyment i kategorię dla całego DataFrame."""
        results = df["text"].apply(self._analyze)
        df["sentiment"]  = results.apply(lambda x: x[0])
        df["confidence"] = results.apply(lambda x: x[1])
        df["category"]   = df["text"].apply(self._detect_category)
        return df

    def predict_single(self, text: str) -> dict:
        sentiment, conf = self._analyze(text)
        category = self._detect_category(text)
        return {
            "text": text,
            "sentiment": sentiment,
            "confidence": conf,
            "category": category,
        }

    def _analyze(self, text: str):
        lower = text.lower()

        neg_score = sum(1 for w in NEG_WORDS if w in lower)
        pos_score = sum(1 for w in POS_WORDS if w in lower)

        # Intensyfikatory
        for word in NEG_INTENSIFIERS:
            if word in lower:
                neg_score += 0.5

        # Wykrzykniki jako sygnał intensywności
        excl = lower.count("!")
        if excl >= 2:
            neg_score += excl * 0.3

        # Caps lock jako sygnał
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if caps_ratio > 0.3:
            neg_score += 1

        diff = neg_score - pos_score
        total = neg_score + pos_score + 1

        if diff > 1.5:
            sentiment = "negatywny"
            conf = min(95, int(60 + (diff / total) * 35 * 10))
        elif diff < -1.5:
            sentiment = "pozytywny"
            conf = min(95, int(60 + (abs(diff) / total) * 35 * 10))
        else:
            sentiment = "neutralny"
            conf = min(85, int(50 + (1 - abs(diff) / max(total, 1)) * 30))

        conf = max(55, min(95, conf))
        return sentiment, conf

    def _detect_category(self, text: str) -> str:
        lower = text.lower()
        for category, keywords in CATEGORY_KEYWORDS.items():
            if category == "inne":
                continue
            if any(kw in lower for kw in keywords):
                return category
        return "inne"


# ─── CRISIS SCORER ────────────────────────────────────────────────────────────

class CrisisScorer:
    """
    Oblicza Crisis Score (0-100) na podstawie wielu sygnałów.

    Składowe:
        Sentyment         0-40 pkt
        Kategoria ryzyka  0-20 pkt
        Intensywność      0-25 pkt  (wykrzykniki, caps, słowa kluczowe)
        Timing            0-15 pkt  (wieczory i weekendy = większe ryzyko)
    """

    def __init__(self, threshold: int = 70):
        self.threshold = threshold

    def score(self, df):
        df["crisis_score"] = df.apply(self._compute, axis=1)
        df["alert_level"]  = df["crisis_score"].apply(self._level)
        return df

    def score_single(self, data: dict) -> dict:
        score = self._compute(data)
        data["crisis_score"] = score
        data["alert_level"]  = self._level(score)
        return data

    def _compute(self, row) -> int:
        score = 0.0
        text      = row.get("text", "")
        sentiment = row.get("sentiment", "neutralny")
        category  = row.get("category", "inne")

        # 1. Sentyment (0-40)
        if sentiment == "negatywny":
            score += 40
        elif sentiment == "neutralny":
            score += 15

        # 2. Kategoria (0-20)
        if category in HIGH_RISK_CATEGORIES:
            score += 20
        elif category == "opóźnienie":
            score += 10

        # 3. Intensywność tekstu (0-25)
        lower = text.lower()
        excl = text.count("!")
        score += min(excl * 2, 10)

        for word in NEG_INTENSIFIERS:
            if word in lower:
                score += 3

        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if caps_ratio > 0.35:
            score += 8

        # Wzmianki o działaniach prawnych
        if any(w in lower for w in ["prawni", "sąd", "pozew", "urząd"]):
            score += 5

        # 4. Timing (0-15) – symulowany na podstawie pola 'date'
        date_val = row.get("date")
        if date_val:
            try:
                if isinstance(date_val, str):
                    dt = datetime.strptime(date_val, "%Y-%m-%d")
                else:
                    dt = date_val
                if dt.hour >= 18 or dt.weekday() >= 5:
                    score += 10
                if dt.weekday() >= 5 and dt.hour >= 18:
                    score += 5
            except Exception:
                pass

        return max(0, min(100, int(round(score))))

    def _level(self, score: int) -> str:
        if score >= self.threshold:
            return "🔴 Wysoki"
        elif score >= self.threshold * 0.6:
            return "🟡 Średni"
        return "🟢 Niski"
