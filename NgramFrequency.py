import requests
import json
from pathlib import Path
from collections import OrderedDict

class NgramAPI:
    def __init__(self, config):
        self.base_url = "https://books.google.com/ngrams/json"
        self.corpus = config["corpus"]
        self.smoothing = config["smoothing"]
        self.timeout = config["timeout"]
        self.cache_path = Path("word_freq.json")
        self.cache = OrderedDict()
        self.max_cache_size = 1000

    def _load_cache(self):
        try:
            if self.cache_path.exists():
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return {}
        except:
            return {}

    def _save_cache(self):
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(dict(self.cache), f)

    def get_ngram_frequency(self, word: str) -> float:
        word = word.lower()
        if word in self.cache:
            return self.cache[word]
        
        if len(self.cache) >= self.max_cache_size:
            self.cache.popitem(last=False)
        
        try:
            response = requests.get(
                f"{self.base_url}?content={word}&corpus={self.corpus}&smoothing={self.smoothing}",
                timeout=self.timeout
            )
            if response.status_code == 200:
                data = response.json()
                if data:
                    freq = sum(data[0]['timeseries'])/len(data[0]['timeseries'])*100
                    self.cache[word] = freq
                    self._save_cache()
                    return freq
            return 0.0
        except Exception as e:
            print(f"API请求失败: {str(e)}")
            return 0.0