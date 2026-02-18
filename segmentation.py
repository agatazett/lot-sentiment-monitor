"""
segmentation.py – Segmentacja pasażerów LOT
"""
import pandas as pd

SEGMENT_KEYWORDS = {
    "✈️ Frequent Flyer": ["frequent","miles","lounge","upgrade","gold","platinum",
        "kolejny raz","znowu lecę","co miesiąc","regularnie","business class",
        "biznes klasa","priority","vip","karta lojalnościowa","punkty"],
    "👨‍👩‍👧 Turysta rodzinny": ["dzieci","dziecko","córka","syn","rodzina","rodzinny",
        "wakacje","urlop","wyjazd","lato","ferie","wózek","maluch","niemowlę"],
    "💼 Podróżny biznesowy": ["biznes","business","służbowy","konferencja","spotkanie",
        "delegacja","klient","kontrakt","praca","firma","meeting","zawodowy","biuro"],
    "🎒 Turysta budżetowy": ["tani","tanio","budżet","najtaniej","promocja","okazja",
        "hostel","plecak","student","studenci","oszczędny","pierwszy raz lecę"],
}

SEGMENT_PROFILES = {
    "✈️ Frequent Flyer":   {"label":"Frequent Flyer","sensitivity":"Bardzo wysoka","priority":"🔴 Krytyczny","color":"#EF4444",
        "desc":"Pasażer regularny, lata kilka razy w miesiącu. Wysoka lojalność ale też wysokie oczekiwania.",
        "actions":["Priorytetowa obsługa reklamacji (max 24h)","Dedykowany opiekun klienta premium","Proaktywny kontakt przy każdym opóźnieniu"]},
    "👨‍👩‍👧 Turysta rodzinny": {"label":"Turysta rodzinny","sensitivity":"Wysoka","priority":"🟠 Wysoki","color":"#F97316",
        "desc":"Podróżuje z dziećmi, raz-dwa razy w roku. Szczególnie wrażliwy na chaos i brak informacji.",
        "actions":["Dedykowana linia obsługi dla rodzin","Automatyczne SMS-y o statusie lotu","Pierwszeństwo przy rebookingu"]},
    "💼 Podróżny biznesowy":{"label":"Podróżny biznesowy","sensitivity":"Wysoka","priority":"🟠 Wysoki","color":"#3B82F6",
        "desc":"Podróżuje służbowo. Ceni punktualność. Opóźnienia mają dla niego realne koszty finansowe.",
        "actions":["Odszkodowanie do 48h bez formalności","Elastyczna polityka zmiany biletów","Raportowanie dla działów HR"]},
    "🎒 Turysta budżetowy": {"label":"Turysta budżetowy","sensitivity":"Średnia","priority":"🟡 Średni","color":"#F59E0B",
        "desc":"Podróżuje okazjonalnie, szuka najtańszych opcji. Aktywny w social mediach.",
        "actions":["Jasna komunikacja kosztów dodatkowych","Program lojalnościowy dla nowych","Responsywny chatbot"]},
    "❓ Nieokreślony":      {"label":"Nieokreślony","sensitivity":"Nieznana","priority":"⚪ Brak danych","color":"#6B7A99",
        "desc":"Profil niemożliwy do określenia na podstawie treści.",
        "actions":["Zbierz więcej danych"]},
}

class PassengerSegmentation:
    def segment(self, df):
        df["segment"] = df["text"].apply(self._classify)
        return df

    def segment_single(self, text):
        return self._classify(text)

    def _classify(self, text):
        lower = text.lower()
        scores = {seg: sum(1 for kw in kws if kw in lower) for seg, kws in SEGMENT_KEYWORDS.items()}
        scores = {k:v for k,v in scores.items() if v>0}
        return max(scores, key=scores.get) if scores else "❓ Nieokreślony"

    def get_profile(self, segment):
        return SEGMENT_PROFILES.get(segment, SEGMENT_PROFILES["❓ Nieokreślony"])

    def summary(self, df):
        if "segment" not in df.columns:
            df = self.segment(df)
        rows = []
        for seg, profile in SEGMENT_PROFILES.items():
            sub = df[df["segment"] == seg]
            if len(sub) == 0:
                continue
            neg_pct = round(len(sub[sub["sentiment"]=="negatywny"])/len(sub)*100,1) if "sentiment" in df.columns else 0
            avg_crisis = round(sub["crisis_score"].mean(),1) if "crisis_score" in sub.columns else 0
            rows.append({"Segment":seg,"Liczba opinii":len(sub),"% negatywnych":f"{neg_pct}%",
                         "Śr. Crisis Score":avg_crisis,"Priorytet":profile["priority"],"Wrażliwość":profile["sensitivity"]})
        return pd.DataFrame(rows)

# Rozszerzenie słowników dla lepszej klasyfikacji polskich tekstów
SEGMENT_KEYWORDS["✈️ Frequent Flyer"].extend([
    "kolejny","znowu","kolejne","kolejna","po raz","czwarty","trzeci","piąty",
    "rok","regularnie","zawsze","zwykle","zazwyczaj","business","klasa biznes"
])
SEGMENT_KEYWORDS["👨‍👩‍👧 Turysta rodzinny"].extend([
    "wyjechaliśmy","polecieliśmy","zostaliśmy","zostałam","zostałem",
    "całą","całe","wszyscy","zabawa","morze","góry","plaża"
])
SEGMENT_KEYWORDS["💼 Podróżny biznesowy"].extend([
    "spotkanie","konferencja","ważne","partner","projekt","prezentacja",
    "straciłem","spóźniłem","nie dotarłem","kosztuje mnie"
])
SEGMENT_KEYWORDS["🎒 Turysta budżetowy"].extend([
    "najtańszy","tańszy","wyjazd","pierwsze","wycieczka","zwiedzanie",
    "byłam","byłem","pojechałam","pojechałem"
])
