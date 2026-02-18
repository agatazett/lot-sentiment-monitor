# ✈️ LOT Polish Airlines – System Predykcji Nastrojów Pasażerów (NLP)

Prototyp systemu wczesnego ostrzegania przed kryzysami wizerunkowymi,
zbudowany na potrzeby pracy dyplomowej.

---

## 📁 Struktura plików

```
lot_nlp/
├── app.py              ← Główna aplikacja Streamlit (dashboard)
├── model.py            ← Model NLP: klasyfikacja sentymentu + Crisis Score
├── data_generator.py   ← Generator polskich opinii (dane syntetyczne)
├── requirements.txt    ← Zależności Python
└── README.md           ← Ten plik
```

---

## 🚀 Uruchomienie (krok po kroku)

### 1. Wymagania
- Python 3.10 lub nowszy
- pip (menedżer pakietów)

### 2. Instalacja zależności

```bash
# Otwórz terminal w folderze lot_nlp/
pip install -r requirements.txt
```

### 3. Uruchomienie dashboardu

```bash
streamlit run app.py
```

Przeglądarka otworzy się automatycznie pod adresem:
**http://localhost:8501**

---

## 🧠 Jak działa model

### Architektura (baseline – reguły słownikowe)
```
Opinia tekstowa
       │
       ▼
┌─────────────────────┐
│   Preprocessing     │  lowercase, usunięcie znaków spec.
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Słownik NEG/POS    │  dopasowanie słów kluczowych
│  + Intensyfikatory  │  wykrzykniki, caps lock
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Klasyfikacja       │  negatywny / neutralny / pozytywny
│  Kategorii          │  opóźnienie / bagaż / obsługa / ...
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Crisis Score       │  0-100 (sentyment + kategoria + intensywność + timing)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Alert / Dashboard  │  🔴 Wysoki / 🟡 Średni / 🟢 Niski
└─────────────────────┘
```

### Formuła Crisis Score
| Składowa            | Max punktów | Opis                                          |
|---------------------|-------------|-----------------------------------------------|
| Sentyment           | 40 pkt      | Negatywny=40, Neutralny=15, Pozytywny=0       |
| Kategoria ryzyka    | 20 pkt      | Odszkodowanie/Komunikacja/Bagaż = max 20 pkt  |
| Intensywność        | 25 pkt      | Wykrzykniki, caps lock, słowa prawnicze       |
| Timing              | 15 pkt      | Wieczory (18+) i weekendy = +10-15 pkt        |

---

## 🔧 Upgrade do prawdziwego modelu NLP

Aby zastąpić reguły słownikowe prawdziwym modelem transformerowym:

### Opcja A – XLM-RoBERTa (wielojęzyczny, szybki)
```python
# W model.py odkomentuj:
from transformers import pipeline

self.pipe = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
)
```

### Opcja B – HerBERT (najlepszy dla polskiego)
```python
from transformers import pipeline

self.pipe = pipeline(
    "text-classification",
    model="allegro/herbert-base-cased"
    # Uwaga: wymaga fine-tuningu na danych sentymentu
)
```

### Opcja C – OpenAI API (najprostszy, płatny)
```python
from openai import OpenAI
client = OpenAI(api_key="sk-...")

def analyze(text):
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":
            f'Oceń sentyment tej opinii o LOT (negatywny/neutralny/pozytywny) '
            f'i podaj kategorię (opóźnienie/bagaż/obsługa/komunikacja/odszkodowanie/inne). '
            f'Odpowiedz w JSON. Opinia: "{text}"'
        }]
    )
    return r.choices[0].message.content
```

---

## 📊 Funkcje dashboardu

| Zakładka           | Funkcja                                                     |
|--------------------|-------------------------------------------------------------|
| 📊 Dashboard       | KPI metryki, trendy sentymentu, kategorie, Crisis Score     |
| 🔬 Analizator      | Analiza pojedynczej opinii z zaleceniami PR                 |
| 📋 Tabela danych   | Pełna tabela z filtrami i eksportem CSV                     |

### Panel boczny (Sidebar)
- Liczba opinii do analizy (50-500)
- Zakres czasowy (7-90 dni)
- Próg alertu Crisis Score (50-90)
- Filtrowanie kategorii
- Odświeżanie danych

---

## 📖 Opis dla pracy dyplomowej

System został zbudowany jako **prototyp proof-of-concept** demonstrujący
architekturę systemu predykcji nastrojów pasażerów LOT Polish Airlines.

**Metodologia:**
- Dane: syntetyczny zbiór 200 polskich opinii (rozkład: 58% neg, 21% neu, 21% poz)
- Model: klasyfikator oparty na analizie leksykalnej z polskim słownikiem sentymentu
- Crisis Score: wieloczynnikowy wskaźnik ryzyka (sentyment, kategoria, intensywność, timing)
- Wizualizacja: dashboard Streamlit z wykresami Plotly

**W środowisku produkcyjnym** system byłby rozszerzony o:
1. Pobieranie danych w czasie rzeczywistym (Twitter API, Google My Business API)
2. Model transformerowy (HerBERT/XLM-RoBERTa) zamiast reguł słownikowych
3. Automatyczne alerty e-mail do zespołu PR
4. Integrację z CRM LOT-u

---

## 🛠️ Technologie

| Warstwa       | Technologia                    |
|---------------|--------------------------------|
| Język         | Python 3.11                    |
| Dashboard     | Streamlit 1.32                 |
| Wykresy       | Plotly 5.18                    |
| Dane          | Pandas 2.0                     |
| Model (prod.) | HerBERT / XLM-RoBERTa          |
| Źródła danych | Twitter API, Google, Trustpilot|

---

*Projekt stworzony na potrzeby pracy dyplomowej.*
*"Wykorzystanie AI do predykcji niezadowolenia pasażerów i zapobiegania kryzysom wizerunkowym LOT Polish Airlines"*
