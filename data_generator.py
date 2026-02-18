"""
data_generator.py – Generator syntetycznych polskich opinii o LOT
=================================================================
Generuje realistyczne dane treningowe i testowe w języku polskim.
Opinie są zróżnicowane pod względem tonu, kategorii i długości.
"""

import random
import pandas as pd
from datetime import datetime, timedelta


# ─── SZABLONY OPINII ─────────────────────────────────────────────────────────

NEGATIVE_OPINIONS = [
    # Opóźnienia
    "Lot opóźniony {delay} godzin! Zero informacji od obsługi, nikt nic nie tłumaczy. Skandal absolutny!",
    "Kolejne opóźnienie LOT-u. Siedzimy już {delay} godziny na lotnisku bez żadnego komunikatu. Hańba!",
    "Lot miał wylecieć o {time}, dopiero po {delay}h poinformowali że jest opóźnienie. Beznadziejne!",
    "Trzecie opóźnienie w ciągu roku z LOT. Tym razem {delay} godziny bez vouchera, bez przeprosin. Nigdy więcej!",
    "Opóźnienie {delay}h i zero rekompensaty. Straciłem ważne spotkanie biznesowe. LOT to katastrofa!",
    "LOT znowu w swoim repertuarze – opóźnienie o {delay} godziny i nikt nawet nie przeprosił pasażerów.",
    "Lot z Warszawy do Londynu opóźniony o {delay}h. Połączenie przesiadkowe przepadło. Dramat!",

    # Bagaż
    "Zaginął mi bagaż na trasie Warszawa-{dest}. Minęły {days} dni i LOT w ogóle nie odpowiada na maile!",
    "Uszkodzona walizka po locie LOT. Reklamacja złożona {days} dni temu – zero odzewu. Skandal!",
    "Bagaż dotarł do {dest} bez mnie. Przez {days} dni nie mogłam uzyskać żadnej informacji od LOT.",
    "Trzecia reklamacja za zaginiony bagaż i cisza ze strony LOT. Ich obsługa klienta to fikcja.",
    "LOT zgubił moją walizkę z ważnymi dokumentami biznesowymi. {days} dni i żadnego kontaktu!",

    # Obsługa
    "Stewardessa była wyjątkowo nieuprzejma gdy poprosiłam o dodatkowy koc. Poczułam się jak problem.",
    "Personel pokładowy ignorował pasażerów przez całą podróż. Zero profesjonalizmu od LOT.",
    "Obsługa naziemna LOT to katastrofa. Chamskie traktowanie pasażerów w {airport}.",
    "Pracownik LOT na lotnisku zachował się skandalicznie wobec starszej pani. Wstyd!",
    "Kelner w samolocie LOT rzucił mi posiłek zamiast podać. Takie traktowanie to nie do przyjęcia.",

    # Komunikacja
    "LOT odwołał lot SMS-em na 2 godziny przed wylotem. Żadnej alternatywy, żadnej pomocy. Chaos totalny!",
    "Strona LOT.com nie działała przez całą dobę. Nie mogłam zrobić check-in i obsługa nic z tym nie robiła.",
    "Informacje o opóźnieniu zmieniały się co godzinę. LOT nie ma pojęcia co się dzieje na ich lotniskach.",
    "Nikt w call center LOT nie odbiera telefonu. Czekałam {delay} godzin na połączenie. Dramat!",
    "LOT poinformował o zmianie trasy lotu e-mailem który dotarł PO ODLOCIE. Niesamowite!",

    # Odszkodowania
    "Żądam odszkodowania zgodnie z EC 261/2004 za odwołany lot! LOT ignoruje moje pisma od {days} dni.",
    "EC 261/2004 daje mi prawo do odszkodowania {amount} EUR, ale LOT odmawia wypłaty bez podstawy prawnej.",
    "Odwołali nasz lot dzień przed ślubem! LOT zaproponował zwrot biletu zamiast odszkodowania. Prawnicy już zaangażowani.",
    "Refund za odwołany lot czeka {days} dni. LOT przelewa pieniądze przez kwartał. Skandaliczne praktyki!",
]

POSITIVE_OPINIONS = [
    "Fantastyczny lot do {dest}! Punktualnie, załoga bardzo miła i profesjonalna. Gorąco polecam LOT!",
    "LOT zaskoczył mnie pozytywnie na trasie Warszawa-{dest}. Jedzenie smaczne, fotele wygodne. Wracam!",
    "Pierwszy raz leciałem business class LOT-em – rewelacja! Obsługa na najwyższym poziomie.",
    "Punktualny start, punktualne lądowanie, bagaż pierwszy z taśmy. LOT się poprawia, super!",
    "Świetna obsługa na pokładzie, miłe stewardesy, pyszne jedzenie. Lot do {dest} był przyjemnością.",
    "LOT naprawdę dba o pasażerów. Lot opóźniony ale dostałem voucher i przeprosiny. Klasa!",
    "Zadowolony z lotu Warszawa-{dest}. Profesjonalna obsługa, czysty samolot, bez problemów. Polecam!",
    "Mile zaskoczony LOT-em po długiej przerwie. Widać że pracują nad jakością. Wracam do Was!",
    "Lot dzieckiem był moją obawą, ale stewardesy były cudowne i pomocne przez całą podróż. Dziękuję LOT!",
    "Bardzo wygodny lot do {dest}. Szersze siedzenia niż u konkurencji, lepsza obsługa. LOT top!",
]

NEUTRAL_OPINIONS = [
    "Lot był w porządku, nic specjalnego ani złego. Dotarłem do {dest} na czas.",
    "Opóźnienie {delay}0 minut – w granicach normy. Jedzenie przeciętne, obsługa OK.",
    "Standardowy lot LOT. Ani super ani tragedia. Pasażer dotarł do celu.",
    "LOT jak LOT – nie najgorszy ani nie najlepszy. Do {dest} dotarłem.",
    "Lot neutralny. Lekkie opóźnienie, bagaż dotarł cały. Nic do dodania.",
    "Średnia obsługa, przeciętne jedzenie. Lot do {dest} bez niespodzianek.",
    "Lot ok. Trochę zimno w samolocie, ale można wytrzymać. {dest} osiągnięty.",
]

DESTINATIONS = ["Londynu", "Nowego Jorku", "Chicago", "Frankfurtu", "Paryża",
                "Amsterdamu", "Berlina", "Barcelony", "Tokio", "Dubaju", "Toronto"]
AIRPORTS = ["Okęciu", "Chopina", "Krakowie", "Gdańsku", "Katowicach", "Wrocławiu"]
AMOUNTS = [250, 400, 600]


def _fill_template(template: str) -> str:
    """Wypełnia szablony opinii losowymi wartościami."""
    return template.format(
        delay=random.randint(2, 8),
        days=random.randint(5, 30),
        dest=random.choice(DESTINATIONS),
        airport=random.choice(AIRPORTS),
        amount=random.choice(AMOUNTS),
        time=f"{random.randint(6,20):02d}:{random.choice(['00','15','30','45'])}",
    )


def generate_sample_data(n: int = 200, days_back: int = 30) -> pd.DataFrame:
    """
    Generuje n syntetycznych opinii pasażerów LOT w języku polskim.

    Rozkład sentymentu (realistyczny dla LOT):
        Negatywny  ~58%
        Neutralny  ~21%
        Pozytywny  ~21%

    Args:
        n:         liczba opinii do wygenerowania
        days_back: zakres dat wstecz od dziś

    Returns:
        DataFrame z kolumnami: text, date, source, likes
    """
    n_neg = int(n * 0.58)
    n_neu = int(n * 0.21)
    n_pos = n - n_neg - n_neu

    records = []
    base_date = datetime.now()

    sources = ["Twitter/X", "Google Reviews", "Trustpilot", "Facebook", "Skytrax"]
    source_weights = [0.40, 0.25, 0.15, 0.15, 0.05]

    def random_date():
        delta = timedelta(
            days=random.randint(0, days_back),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )
        return (base_date - delta).strftime("%Y-%m-%d")

    # Negatywne
    for _ in range(n_neg):
        tmpl = random.choice(NEGATIVE_OPINIONS)
        records.append({
            "text":   _fill_template(tmpl),
            "date":   random_date(),
            "source": random.choices(sources, source_weights)[0],
            "likes":  random.randint(0, 200),
        })

    # Neutralne
    for _ in range(n_neu):
        tmpl = random.choice(NEUTRAL_OPINIONS)
        records.append({
            "text":   _fill_template(tmpl),
            "date":   random_date(),
            "source": random.choices(sources, source_weights)[0],
            "likes":  random.randint(0, 30),
        })

    # Pozytywne
    for _ in range(n_pos):
        tmpl = random.choice(POSITIVE_OPINIONS)
        records.append({
            "text":   _fill_template(tmpl),
            "date":   random_date(),
            "source": random.choices(sources, source_weights)[0],
            "likes":  random.randint(0, 150),
        })

    random.shuffle(records)
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df
