"""
brand24_connector.py – Konektor Brand24 API + Twitter/X API
"""
import os, random, pandas as pd
from datetime import datetime, timedelta

try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class TwitterConnector:
    def __init__(self, bearer_token=None):
        self.bearer_token = bearer_token or os.getenv("TWITTER_BEARER_TOKEN","")
        self.connected = False
        if self.bearer_token and TWEEPY_AVAILABLE:
            try:
                self.client = tweepy.Client(bearer_token=self.bearer_token, wait_on_rate_limit=False)
                self.connected = True
            except Exception:
                pass

    def fetch(self, max_results=50):
        if not self.connected:
            return self._demo(max_results)
        try:
            q = '(@LOTairlines OR "LOT Polish Airlines") lang:pl -is:retweet'
            resp = self.client.search_recent_tweets(query=q, max_results=min(max_results,100),
                tweet_fields=["created_at","text","public_metrics"])
            if not resp.data:
                return self._demo(max_results)
            records = []
            for t in resp.data:
                records.append({"text":t.text,
                    "date": t.created_at.strftime("%Y-%m-%d") if t.created_at else datetime.now().strftime("%Y-%m-%d"),
                    "source":"Twitter/X",
                    "likes": t.public_metrics.get("like_count",0) if t.public_metrics else 0,
                    "retweets": t.public_metrics.get("retweet_count",0) if t.public_metrics else 0,
                    "reach": 0})
            return pd.DataFrame(records)
        except Exception:
            return self._demo(max_results)

    def _demo(self, n):
        tweets = [
            "Lot opóźniony 3h @LOTairlines i nikt nic nie mówi!! #skandal",
            "@LOTairlines gdzie jest mój bagaż?? 5 dni bez informacji!!!",
            "Fantastyczny lot z @LOTairlines do NYC, polecam biznes klasę!",
            "@LOTairlines odwołali lot dzień przed wylotem. Żądam EC 261!",
            "Stewardessa @LOTairlines była wyjątkowo pomocna ❤️",
            "@LOTairlines nie odbieracie telefonów od 3h. Katastrofa!",
            "Kolejne opóźnienie @LOTairlines, 4. raz w tym roku #LOT",
            "Zaginął bagaż @LOTairlines WAW-LHR. Dezorganizacja totalna.",
            "Super obsługa @LOTairlines WAW-ORD. Punktualnie i czysto!",
            "@LOTairlines zwrot za odwołany lot czeka 45 dni. Gdzie kasa?",
            "Lot @LOTairlines ok, jedzenie przeciętne, dotarłem na czas.",
            "@LOTairlines brak informacji o gate! Chaos na lotnisku!",
            "Business class @LOTairlines – niesamowite doświadczenie! Polecam!",
            "@LOTairlines aplikacja nie działa od tygodnia. Kiedy naprawa?",
            "Miły personel @LOTairlines, lot do Paryża bez zarzutu.",
        ]
        base = datetime.now()
        records = [{"text": random.choice(tweets),
                    "date": (base - timedelta(days=random.randint(0,14), hours=random.randint(0,23))).strftime("%Y-%m-%d"),
                    "source": "Twitter/X (demo)",
                    "likes": random.randint(0,150),
                    "retweets": random.randint(0,40),
                    "reach": random.randint(100,5000)} for _ in range(n)]
        return pd.DataFrame(records)


class Brand24Connector:
    SOURCES = ["Twitter","Facebook","Instagram","YouTube","Fora internetowe","Portale newsowe","Blogi","TikTok"]

    def __init__(self, api_token=None):
        self.api_token = api_token or os.getenv("BRAND24_API_TOKEN","")
        self.connected = bool(self.api_token and len(self.api_token) > 10)

    def fetch(self, days_back=30, max_results=200):
        if self.connected:
            return self._fetch_real(days_back, max_results)
        return self._fetch_simulated(days_back, max_results)

    def _fetch_real(self, days_back, max_results):
        try:
            import requests
            headers = {"Authorization": f"Bearer {self.api_token}"}
            params = {"date_from": (datetime.now()-timedelta(days=days_back)).strftime("%Y-%m-%d"),
                      "per_page": min(max_results,1000)}
            resp = requests.get("https://app.brand24.com/api/v3/mentions",
                                headers=headers, params=params, timeout=10)
            if resp.status_code != 200:
                return self._fetch_simulated(days_back, max_results)
            data = resp.json().get("mentions",[])
            records = [{"text": m.get("description",""), "date": m.get("published_at","")[:10],
                        "source": m.get("source","Brand24"),
                        "likes": m.get("facebook_statistics",{}).get("likes",0),
                        "retweets": m.get("twitter_statistics",{}).get("retweets",0),
                        "reach": m.get("estimated_reach",0),
                        "url": m.get("url",""), "author": m.get("author_name",""),
                        "brand24_sentiment": m.get("sentiment","neutral")} for m in data]
            return pd.DataFrame(records)
        except Exception:
            return self._fetch_simulated(days_back, max_results)

    def _fetch_simulated(self, days_back, max_results):
        from data_generator import generate_sample_data
        df = generate_sample_data(max_results, days_back)
        df["source"]   = [random.choice(self.SOURCES) for _ in range(len(df))]
        df["reach"]    = [random.randint(100, 50000) for _ in range(len(df))]
        df["url"]      = [f"https://example.com/post/{random.randint(10000,99999)}" for _ in range(len(df))]
        df["author"]   = [f"użytkownik_{random.randint(100,9999)}" for _ in range(len(df))]
        df["retweets"] = [random.randint(0, 50) for _ in range(len(df))]
        df["brand24_sentiment"] = [random.choices(["negative","neutral","positive"],[0.58,0.21,0.21])[0] for _ in range(len(df))]
        return df

    @staticmethod
    def from_csv(filepath):
        df = pd.read_csv(filepath, encoding="utf-8-sig")
        col_map = {"Date":"date","Content":"text","Source":"source","Likes":"likes",
                   "Reach":"reach","URL":"url","Author":"author","Sentiment":"brand24_sentiment",
                   "Data":"date","Treść":"text","Źródło":"source","Polubienia":"likes","Zasięg":"reach"}
        df = df.rename(columns={k:v for k,v in col_map.items() if k in df.columns})
        for col in ["text","date","source","likes"]:
            if col not in df.columns:
                df[col] = "" if col in ["text","source","date"] else 0
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df = df.dropna(subset=["text"])
        df["text"] = df["text"].astype(str)
        return df


class FacebookGroupsConnector:
    GROUPS = ["LOT Polish Airlines - Opinie","Odszkodowania za loty EC261","Pasażerowie LOT"]
    POSTS = [
        "Czy ktoś dostał odszkodowanie od LOT? Czekam już 2 miesiące...",
        "Poleciałam LOT-em do Londynu, obsługa rewelacyjna! Coś się zmienia?",
        "LOT odwołał lot w ostatniej chwili. Zostaliśmy z dziećmi na lotnisku. Koszmar!",
        "Czy ktoś zna działający numer do LOT? Call center nieosiągalne od rana...",
        "Bagaż LOT dotarł 3 dni po mnie. PIR złożony, 2 tygodnie i cisza. Co robić?",
        "Poleciałam biznes klasą LOT WAW-JFK. Cena wysoka ale jakość adekwatna. Polecam!",
        "UWAGA! LOT zmienił mi lot bez powiadomienia. Dowiedziałem się przez przypadek.",
        "Czy LOT honoruje EC 261? Opóźnienie 4h i odmówili bez podstawy.",
        "Świetna inicjatywa LOT z nowymi trasami! W końcu coś pozytywnego.",
        "Załoga LOT na WAW-JFK była naprawdę profesjonalna. Miłe zaskoczenie!",
    ]

    def fetch(self, n=100, days_back=30):
        base = datetime.now()
        records = [{"text": random.choice(self.POSTS),
                    "date": (base - timedelta(days=random.randint(0,days_back), hours=random.randint(0,23))).strftime("%Y-%m-%d"),
                    "source": f"Facebook: {random.choice(self.GROUPS)}",
                    "likes": random.randint(0,300), "retweets": 0,
                    "reach": random.randint(200,10000),
                    "author": f"fb_user_{random.randint(100,9999)}"} for _ in range(n)]
        return pd.DataFrame(records)
