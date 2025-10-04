# Grammar Checker API 
import requests

class LanguageToolChecker:
    def __init__(self):
        self.endpoint = "https://api.languagetoolplus.com/v2/check"

    def has_grammar_error(self, sentence: str) -> bool:
        try:
            response = requests.post(
                self.endpoint,
                data={
                    "text": sentence,
                    "language": "en-US"
                }
            )
            result = response.json()
            return len(result.get("matches", [])) > 0
        except Exception as e:
            print(f"[Grammar API Error] {e}")
            return False  # 如果 API 失败，默认当做“无错”