import os
import re
import yaml
import math
import random
import time
import spacy
import torch
import requests
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from Grammar_Check import LanguageToolChecker
from NgramFrequency import NgramAPI


class MazeGenerator:
    # Words to avoid
    FORBIDDEN_WORDS = [
        '.', '."', '-', ' ', ' ', ' ', '<|endoftext|>', 'to', 'on', 'doesn', r"'ll", "not", r"'s",
        've', "'", 're', "'d", ',', 'isn', '-', ',', '!', '?', ';', ':', "s", '"', '…', '(', ')', '[',
        ']', '{', '}', '-', '_', '=', '+', '*', '/', '\\', '|', '<', '>', '@', '#', '$', '%', '^', '&',
        "a","an","the","this","that","these","those","and","or","but","to","of","in","on","at","if",
        "for","with","from","as","than","so","she","he","it","they","there","here","me","him","her",
        "them","us","i","you","we","example","very"
    ]

    # —— Rule prompt (system) ——
    BASE_PROMPT = """You are a grammar expert following Veronica Boyce's Maze methodology. Generate words that would make sentences ungrammatical.

You are selecting ONE distractor word for a Maze task.

Constraints (all MUST hold):
1) Output exactly one lowercase English WORD (regex: ^[a-z]{3,12}$). No digits, hyphens, apostrophes, spaces, or punctuation.
2) The word must be a common, high-frequency everyday English word (avoid archaic, technical jargon, slang, brand names, or nonce formations).
3) Part of speech: return a {POS_LABEL} ONLY, in base form:
   - VERB → bare infinitive (e.g., run, bake; NOT runs/baked/running)
   - NOUN → common singular (e.g., table; NOT tables/table’s)
   - ADJ  → base form (e.g., bright; NOT brighter/brightest)
   No proper nouns or pronouns; no function words.
4) No repeats or morphological relatives:
   - Do NOT repeat any token from S (case-insensitive).
   - Do NOT return any word that shares the same lemma/morphological family as any token in S or any item in USED.
     (Example: if S contains “photo/photos/photography”, avoid photo/photos/photograph/photographic/photography.)
5) Grammar-breaking requirement:
   Insert the word at index i of S (0-based, space-delimited). If the resulting sentence is still grammatical in standard English, REJECT and choose another word.
   Heuristics to AVOID grammatical outputs:
   - If the left token is a determiner (a/an/the/this/that) and the right token is a noun, do NOT output an ADJ or NOUN (would form a grammatical NP).
   - If the position is a subject pronoun/proper noun slot followed by an auxiliary/verb, return a VERB (not a noun/adj).
   - Avoid choices that create standard adjective–noun or adverb–verb collocations.
6) Diversity: the word must NOT belong to the same lemma as any item in USED (already selected distractors in this sentence).

Return strictly:
<ans>word</ans>

Format: For each position i, output:
<ans_i>word1,word2,word3,word4,word5,word6,word7,word8,word9,word10</ans_i>

Example:
<ans_0>elephant,piano,blue,fast,big,red,small,hot,cold,new</ans_0>
<ans_1>guitar,car,book,house,tree,water,fire,wind,earth,sky</ans_1>
<ans_2>violin,truck,table,ocean,forest,mountain,desert,river,lake,valley</ans_2>
"""

    def __init__(self, config_path="config.yaml"):
        with open("config.json", "r") as f:
            self.wordfreq = json.load(f)
        self.BASE_PROMPT = self.config["BASE_PROMPT"]
        self.min_wordfreq = self.config.get("filters", {}).get("min_wordfreq", 5e-6)
        self._validate_config()

        # Support multiple backends
        self.backend = self.config["model"].get("backend", "GROQ_API").upper()
        
        # Model configuration
        self.model_id = self.config["model"].get("name", "llama-3.1-8b-instant")
        self.local_model_path = self.config["model"].get("local_model_path", None)
        
        # Initialize model
        if self.backend == "LOCAL":
            self._init_local_model()
        elif self.backend == "HF_API":
            self._init_hf_api_model()
        elif self.backend == "GROQ_API":
            self._init_groq_api_model()
        else:
            raise RuntimeError(f"不支持的backend: {self.backend}。支持 LOCAL, HF_API, 或 GROQ_API")

        # Other components 
        self.surprisal_threshold = self.config["model"].get("surprisal_threshold", 21)
        self.similarity_threshold = 0.7
        self.nlp = spacy.load("en_core_web_md")
        self.ngram_api = NgramAPI(self.config["api"]["ngram"])
        self.grammar_checker = LanguageToolChecker()

        # Improved: add word tracking mechanism to prevent repetition
        self.used_words_in_sentence = set()
        self.word_frequency_tracker = {}  # Track global word frequency to avoid excessive repetition
        
        self.verbose = True  # Open print debugging scores

    def _init_local_model(self):
        """初始化本地Llama2模型"""
        print("正在加载本地Llama2模型...")
        
        # Determine model path
        model_path = self.local_model_path if self.local_model_path and os.path.exists(self.local_model_path) else self.model_id
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            
            # Set pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"Local model loaded successfully: {model_path}")
            
        except Exception as e:
            print(f"Local model loaded failed: {e}")
            print("Trying to download model from HuggingFace...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"Model downloaded from HuggingFace successfully: {self.model_id}")
            except Exception as e2:
                raise RuntimeError(f"Failed to load model: {e2}")

    def _init_hf_api_model(self):
        """Initialize HuggingFace API model"""
        # Directly hardcode token (no longer read environment variables)
        token = ""  # <<< Replace with your Hugging Face token
        if not token or not token.startswith("hf_"):
            raise RuntimeError("请在脚本中把 token 字符串替换成你的 Hugging Face token（以 hf_ 开头）。")
        self._token = token

        # —— Initialize HF Inference API client (chat interface) ——
        from huggingface_hub import InferenceClient
        self.hf = InferenceClient(model=self.model_id, token=token)

    def _init_groq_api_model(self):
        """Initialize Groq API model"""
        # Groq API key
        self.groq_api_key = "gsk_your_groq_api_key_here"  # <<< Replace with your Groq API key
        self.groq_base_url = "Please use you Groq API base URL here"  # <<< Replace with your Groq API base URL
        
        if self.groq_api_key == "gsk_your_groq_api_key_here":
            print("⚠️  Warning: Groq API key not set. Please replace the placeholder with your actual Groq API key.")
            print("Temporarily using HuggingFace API...")
            self._init_hf_api_model()
            self.backend = "HF_API"
            return
        
        print(f"✓ Groq API initialized successfully, using model: {self.model_id}")

    # ================== Basic ==================

    def _load_config(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _validate_config(self):
        need = [
            ("model", dict),
            ("model.name", (str,)),
            ("model.generation", dict),
            ("model.generation.top_k", int),
            ("model.generation.num_return_sequences", int),
            ("api", dict),
            ("api.ngram", dict),
            ("api.ngram.corpus", (str, int, dict)),
        ]
        cfg = self.config

        def _get(path):
            cur = cfg
            for p in path.split("."):
                if p not in cur:
                    raise ValueError(f"Missing config key: {path}")
                cur = cur[p]
            return cur

        for key, typ in need:
            val = _get(key)
            if not isinstance(val, typ):
                raise ValueError(f"Config key `{key}` expects {typ}, got {type(val)}")

    def _validate_input(self, sentence: str):
        if not sentence or not sentence.strip():
            raise ValueError("input cannot be empty")
        if len(sentence.split()) < 2:
            raise ValueError("at least two words are required in the input sentence")

    # ================== Main process (generate one distractor word for each position in the sentence, set desired_pos for each position) ==================

    def generate_full_maze(self, sentence: str) -> dict:
        """
        Generate one distractor word for each position in the sentence.
        Set desired_pos for each position.
        """
        self._validate_input(sentence)
        self.used_words_in_sentence = set()

        tokens = sentence.split()
        pairs = []
        for idx, tok in enumerate(tokens):
            if idx == 0:
                pairs.append(self._create_first_pair(tok))
                continue

            if self._is_punct_only(tok):
                pairs.append({"position": idx, "correct": tok, "distractor": "x-x-x"})
                continue

            target_pos = self._get_word_pos(tok)
            desired_pos = self._choose_desired_pos(target_pos)

            cand = self._generate_one_candidate_api(sentence, idx, len(tokens), desired_pos=desired_pos)
            if not self._is_good_word(cand):
                cand = self._random_corpus_word(not_equal_to=tok)

            pairs.append({"position": idx, "correct": tok, "distractor": cand})
        return {"original": sentence, "maze_pairs": pairs}

    def generate_full_maze_stream(self, sentence: str):
        """
        Compatible with the original streaming interface: generate one distractor word for each position in the sentence, set desired_pos for each position.
        """
        self._validate_input(sentence)
        self.sentence_lexicon = set(re.findall(r"[A-Za-z'-]+", sentence.lower()))
        self.used_words_in_sentence = set()

        tokens = sentence.split()
        yield self._create_first_pair(tokens[0])
        for idx in range(1, len(tokens)):
            tok = tokens[idx]

            if self._is_punct_only(tok):
                yield {"position": idx, "correct": tok, "distractor": "x-x-x"}
                continue

            target_pos = self._get_word_pos(tok)
            desired_pos = self._choose_desired_pos(target_pos)

            cand = self._generate_one_candidate_api(sentence, idx, len(tokens), desired_pos=desired_pos)
            if not self._is_good_word(cand):
                cand = self._random_corpus_word(not_equal_to=tok)

            yield {"position": idx, "correct": tok, "distractor": cand}

    # ================== One-time candidate generation (API, chat_completion) ==================
    # (Keep: if you need to switch back to the whole sentence pool, you can continue using it)

    def _generate_all_candidates_api(self, sentence: str, T: int) -> list:
        if self.backend == "LOCAL":
            return self._generate_all_candidates_local(sentence, T)
        elif self.backend == "HF_API":
            return self._generate_all_candidates_hf_api(sentence, T)
        elif self.backend == "GROQ_API":
            return self._generate_all_candidates_groq_api(sentence, T)
        else:
            raise RuntimeError(f"Unsupported backend: {self.backend}")

    def _generate_all_candidates_local(self, sentence: str, T: int) -> list:
        user_prompt = f'S = "{sentence}"\nT = {T}\nReturn exactly T lines as specified.'
        full_prompt = f"{self.BASE_PROMPT}\n\n{user_prompt}"
        tries, last_err = 4, None
        for _ in range(tries):
            try:
                inputs = self.tokenizer.encode(full_prompt, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=max(64, 12 * T),
                        temperature=self.config["model"]["generation"].get("temperature", 0.8),
                        do_sample=self.config["model"]["generation"].get("do_sample", True),
                        top_k=self.config["model"]["generation"].get("top_k", 50),
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                text = generated_text[len(full_prompt):].strip()
                ans = self._parse_ans_lines(text, T)
                return self._backfill_missing(sentence, ans)
            except Exception as e:
                last_err = e
                time.sleep(3)
        raise last_err

    def _generate_all_candidates_hf_api(self, sentence: str, T: int) -> list:
        user_prompt = f'S = "{sentence}"\nT = {T}\nReturn exactly T lines as specified.'
        tries, last_err = 4, None
        for _ in range(tries):
            try:
                resp = self.hf.chat_completion(
                    messages=[
                        {"role": "system", "content": self.BASE_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max(200, 30 * T),
                    temperature=0.8,
                )
                text = resp.choices[0].message["content"]
                ans = self._parse_ans_lines(text, T)
                return self._backfill_missing(sentence, ans)
            except Exception as e:
                last_err = e
                time.sleep(3)
        raise last_err

    def _generate_all_candidates_groq_api(self, sentence: str, T: int) -> list:
        user_prompt = f'S = "{sentence}"\nT = {T}\nReturn exactly T lines as specified.'
        tries, last_err = 4, None
        for _ in range(tries):
            try:
                headers = {
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": self.model_id,
                    "messages": [
                        {"role": "system", "content": self.BASE_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": max(200, 30 * T),
                    "temperature": 0.8,
                    "stream": False
                }
                response = requests.post(self.groq_base_url, headers=headers, json=data)
                response.raise_for_status()
                result = response.json()
                text = result["choices"][0]["message"]["content"]
                ans = self._parse_ans_lines(text, T)
                return self._backfill_missing(sentence, ans)
            except Exception as e:
                last_err = e
                time.sleep(3)
        raise last_err

    def _parse_ans_lines(self, text: str, T: int) -> list:
        if self.verbose:
            print(f"Parsing text: {text[:200]}...")
        candidate_pools = [None] * T
        pairs = re.findall(r"<ans_(\d+)>\s*([^<]+)\s*</ans_\1>", text, flags=re.I)
        if self.verbose:
            print(f"Found {len(pairs)} matching items")
        for idx_str, words_str in pairs:
            try:
                i = int(idx_str)
                if 0 <= i < T:
                    candidates = [w.strip().lower() for w in words_str.split(',')]
                    valid_candidates = [w for w in candidates if self._is_good_word(w)]
                    if valid_candidates:
                        candidate_pools[i] = valid_candidates
                        if self.verbose:
                            print(f"Position {i}: foun {len(valid_candidates)}matching items")
            except Exception as e:
                if self.verbose:
                    print(f"error in parsing {idx_str} : {e}")
                continue

        final_ans = []
        for i in range(T):
            if candidate_pools[i] is not None and len(candidate_pools[i]) > 0:
                best_word = self._select_least_likely_word(candidate_pools[i])
                if self.verbose:
                    print(f"Position {i}: Selected '{best_word}' from {len(candidate_pools[i])} candidates")
                final_ans.append(best_word)
            else:
                if self.verbose:
                    print(f"Position {i}: No valid candidates found")
                final_ans.append("NO_VALID_DISTRACTOR")
        return final_ans

    def _select_least_likely_word(self, candidates: list, desired_pos: str | None = None) -> str:
        """
        add on desired_pos bonus to improbability score
        """
        if not candidates:
            return "NO_VALID_DISTRACTOR"
        filtered_candidates = []
        for word in candidates:
            if word.lower() not in self.used_words_in_sentence:
                global_count = self.word_frequency_tracker.get(word.lower(), 0)
                if global_count < 3:
                    filtered_candidates.append(word)
        if not filtered_candidates:
            filtered_candidates = candidates
        if len(filtered_candidates) == 1:
            selected_word = filtered_candidates[0]
            self.used_words_in_sentence.add(selected_word.lower())
            self.word_frequency_tracker[selected_word.lower()] = self.word_frequency_tracker.get(selected_word.lower(), 0) + 1
            return selected_word

        scores = []
        for word in filtered_candidates:
            score = self._calculate_improbability_score(word)
            if desired_pos:
                try:
                    pos = self._get_word_pos(word)
                    if pos == desired_pos:
                        score += 3.0  # matching desired POS bonus
                except Exception:
                    pass
            scores.append((word, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        selected_word = scores[0][0]
        self.used_words_in_sentence.add(selected_word.lower())
        self.word_frequency_tracker[selected_word.lower()] = self.word_frequency_tracker.get(selected_word.lower(), 0) + 1
        return selected_word

    def _calculate_improbability_score(self, word: str) -> float:
        score = 0.0
        try:
            freq = self.ngram_api.get_ngram_frequency(word)
            if freq > 0:
                score += 1.0 / (freq + 0.001)
            else:
                score += 10.0
        except:
            score += 5.0
        if len(word) < 3:
            score += 2.0
        elif len(word) > 10:
            score += 1.0
        if '-' in word or "'" in word:
            score += 1.5
        import random
        score += random.uniform(0, 1)
        return score



    def _is_pos(self, word: str, pos: str) -> bool:
        if not word or not pos: return False
        try:
            doc = self.nlp(word)
            return len(doc) > 0 and doc[0].pos_ == pos
        except Exception:
            return False

    def _random_corpus_word_by_pos(self, desired_pos: str | None) -> str:
    # pick from corpus with desired POS
        try:
            if desired_pos and hasattr(self, 'ngram_api') and hasattr(self.ngram_api, 'cache') and self.ngram_api.cache:
                cands = [w for w, f in self.ngram_api.cache.items()
                     if f < 0.1 and self._is_good_word(w) and self._is_pos(w, desired_pos)]
                if cands:
                    w = random.choice(cands)
                    self.used_words_in_sentence.add(w.lower())
                    self.word_frequency_tracker[w.lower()] = self.word_frequency_tracker.get(w.lower(), 0) + 1
                    return w
        except Exception:
            pass

    
        try:
            label = {"NOUN": "noun", "VERB": "verb", "ADJ": "adjective", "ADV": "adverb"}.get(desired_pos)
            if label:
                if self.backend == "LOCAL":
                    return self._generate_fallback_word_local(f"Return ONE {label} only.")
                elif self.backend == "HF_API":
                    return self._generate_fallback_word_hf_api(f"Return ONE {label} only.")
                elif self.backend == "GROQ_API":
                    return self._generate_fallback_word_groq_api(f"Return ONE {label} only.")
        except Exception:
            pass
        return "placeholder"

    # ================== （API + random generation） ==================

    def _backfill_missing(self, sentence: str, ans_list: list) -> list:
        tokens = sentence.split()
        T = len(tokens)
        filled = ans_list[:]
        for i in range(T):
            if filled[i] != "NO_VALID_DISTRACTOR":
                continue
            try:
                # aligning expectation score and POS
                target_pos = self._get_word_pos(tokens[i])
                desired_pos = self._choose_desired_pos(target_pos)
                w = self._generate_one_candidate_api(sentence, i, T, desired_pos=desired_pos)
                if self._is_good_word(w):
                    filled[i] = w
            except Exception:
                pass
        for i in range(T):
            if filled[i] == "NO_VALID_DISTRACTOR":
                target = tokens[i]
                avoid_pos = None
                try:
                    avoid_pos = self._get_word_pos(target)
                except Exception:
                    pass
                w = self._random_corpus_word(avoid_pos=avoid_pos, not_equal_to=target)
                if self.verbose:
                    print(f"Position {i}: Fallback word '{w}' for target '{target}'")
                filled[i] = w if self._is_good_word(w) else "placeholder"
        return filled

    def _generate_one_candidate_api(self, sentence: str, idx: int, T: int, desired_pos: str | None = None) -> str:
        MAX_TRIES = 3
        """
        New function to return pos （NOUN/VERB/ADJ/ADV）。
        """
        for _ in range(MAX_TRIES):
            if self.backend == "LOCAL":
                return self._generate_one_candidate_local(sentence, idx, T, desired_pos=desired_pos)
            elif self.backend == "HF_API":
                return self._generate_one_candidate_hf_api(sentence, idx, T, desired_pos=desired_pos)
            elif self.backend == "GROQ_API":
                return self._generate_one_candidate_groq_api(sentence, idx, T, desired_pos=desired_pos)
            else:
                raise RuntimeError(f"不支持的backend: {self.backend}")
            
            if self._is_good_word(w) and (not desired_pos or self._is_pos(w, desired_pos)):
                return w
            
        fb = self._random_corpus_word_by_pos(desired_pos=desired_pos)
        if self._is_good_word(fb) and (not desired_pos or self._is_pos(fb, desired_pos)):
            return fb
        return self._random_corpus_word()

    def _generate_one_candidate_local(self, sentence: str, idx: int, T: int, desired_pos: str | None = None) -> str:
        pos_label = {"NOUN":"noun","VERB":"verb","ADJ":"adjective","ADV":"adverb"}.get(desired_pos, "content word")
        system_msg = ("You are selecting ONE distractor word for a Maze task.\n\n"
    "Constraints (all MUST hold):\n"
    "1) Output exactly one lowercase English WORD (regex: ^[a-z]{3,12}$). No digits, hyphens, apostrophes, spaces, or punctuation.\n"
    "2) The word must be a common, high-frequency everyday English word (avoid archaic, technical jargon, slang, brand names, or nonce formations).\n"
    f"3) Part of speech: return a {pos_label} ONLY, in base form (verb bare infinitive; noun singular; adjective positive degree). No proper nouns or pronouns; no function words.\n"
    "4) No repeats or morphological relatives of any token in S or any item in USED.\n"
    "5) Grammar-breaking requirement: when the word is inserted at index i in S (0-based, space-delimited), the sentence must become ungrammatical. Avoid choices that complete DET+(ADJ)+N or standard ADJ–N/ADV–V patterns.\n\n"
    "Return strictly:\n<ans>word</ans>"
)
        
        user_msg = f'S = "{sentence}"\nT = {T}\ni = {idx}\nYour answer:'
        
        full_prompt = f"{system_msg}\n\n{user_msg}"
        try:
            inputs = self.tokenizer.encode(full_prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=6,
                    temperature=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            text = generated_text[len(full_prompt):].strip()
            m = re.search(r"<ans>\s*([A-Za-z'-]+)\s*</ans>", text)
            if m:
                return m.group(1).lower()
            toks = re.findall(r"[A-Za-z'-]+", text)
            return (toks[-1].lower() if toks else "NO_VALID_DISTRACTOR")
        except Exception:
            return "NO_VALID_DISTRACTOR"

    def _generate_one_candidate_hf_api(self, sentence: str, idx: int, T: int, desired_pos: str | None = None) -> str:
        system_msg = (
            "Return 50 real English words (1–3 tokens each) that make sentence S UNGRAMMATICAL "
            "if inserted exactly at index i (0-based, space-delimited). "
            "No digits, no punctuation-only, no function words/pronouns. "
            f"Prioritize returning {desired_pos or 'NOUN/VERB/ADJ'} words. "
            "Do NOT use 'ans' or similar placeholder words. "
            "Output strictly as: <ans>word1,word2,word3,...,word50</ans>"
        )
        user_msg = f'S = "{sentence}"\nT = {T}\ni = {idx}\nYour answer:'

        resp = self.hf.chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=100,  # contains 50 -100 words
            temperature=0.9,
        )
        text = resp.choices[0].message["content"]
        m = re.search(r"<ans>\s*([^<]+)\s*</ans>", text)
        if m:
            candidates_str = m.group(1)
            candidates = [w.strip().lower() for w in candidates_str.split(',')]
            valid_candidates = [w for w in candidates if self._is_good_word(w)]
            if valid_candidates:
                return self._select_least_likely_word(valid_candidates, desired_pos=desired_pos)
        toks = re.findall(r"[A-Za-z'-]+", text)
        if toks:
            return toks[-1].lower()
        return "NO_VALID_DISTRACTOR"

    def _generate_one_candidate_groq_api(self, sentence: str, idx: int, T: int, desired_pos: str | None = None) -> str:
        system_msg = (
            "Generate 50 words that would make the sentence ungrammatical if inserted at the specified position. "
            "Use real English words only. Avoid function words. "
            f"Return {desired_pos or 'NOUN/VERB/ADJ'} preferably. "
            "Output format: <ans>word1,word2,word3,word4,word5,word6,word7,word8,word9,word10</ans>"
        )
        user_msg = f'S = "{sentence}"\nT = {T}\ni = {idx}\nYour answer:'

        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                "max_tokens": 100,
                "temperature": 0.9,
                "stream": False
            }
            response = requests.post(self.groq_base_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            m = re.search(r"<ans>\s*([^<]+)\s*</ans>", text)
            if m:
                candidates_str = m.group(1)
                candidates = [w.strip().lower() for w in candidates_str.split(',')]
                valid_candidates = [w for w in candidates if self._is_good_word(w)]
                if valid_candidates:
                    return self._select_least_likely_word(valid_candidates, desired_pos=desired_pos)
            toks = re.findall(r"[A-Za-z'-]+", text)
            if toks:
                return toks[-1].lower()
            return "NO_VALID_DISTRACTOR"
        except Exception:
            return "NO_VALID_DISTRACTOR"

    # —— API driven backup ——
    def _random_corpus_word(self, avoid_pos: str | None = None, not_equal_to: str | None = None, max_trials: int = 80) -> str:
        def ok(w: str) -> bool:
            if not w or w in self.FORBIDDEN_WORDS: return False
            if not re.fullmatch(r"[A-Za-z'-]+", w): return False
            if not_equal_to and w.lower() == not_equal_to.lower(): return False
            if w.lower() in self.used_words_in_sentence: return False
            global_count = self.word_frequency_tracker.get(w.lower(), 0)
            if global_count >= 3: return False
            try:
                if self.ngram_api.get_ngram_frequency(w) <= 0.0:
                    return False
            except Exception:
                return False
            return True

        try:
            if hasattr(self.ngram_api, 'cache') and self.ngram_api.cache:
                low_freq_words = [w for w, freq in self.ngram_api.cache.items() 
                                  if freq < 0.1 and ok(w)]
                if low_freq_words:
                    selected = random.choice(low_freq_words)
                    self.used_words_in_sentence.add(selected.lower())
                    self.word_frequency_tracker[selected.lower()] = self.word_frequency_tracker.get(selected.lower(), 0) + 1
                    return selected
        except Exception:
            pass

        for attempt in range(max_trials):
            try:
                api_word = self._generate_fallback_word_via_api(attempt)
                if api_word and ok(api_word):
                    self.used_words_in_sentence.add(api_word.lower())
                    self.word_frequency_tracker[api_word.lower()] = self.word_frequency_tracker.get(api_word.lower(), 0) + 1
                    return api_word
            except Exception:
                continue
        try:
            final_word = self._generate_emergency_fallback_word()
            if final_word and ok(final_word):
                self.used_words_in_sentence.add(final_word.lower())
                self.word_frequency_tracker[final_word.lower()] = self.word_frequency_tracker.get(final_word.lower(), 0) + 1
                return final_word
        except Exception:
            pass
        return "placeholder"

    def _generate_fallback_word_via_api(self, attempt: int = 0) -> str:
        strategies = [
            "Generate a random English noun that would be ungrammatical in most contexts.",
            "Generate a random English verb that would be ungrammatical in most contexts.", 
            "Generate a random English adjective that would be ungrammatical in most contexts.",
            "Generate a random English animal name that would be ungrammatical in most contexts.",
            "Generate a random English object name that would be ungrammatical in most contexts.",
            "Generate a random English color name that would be ungrammatical in most contexts.",
            "Generate a random English food name that would be ungrammatical in most contexts.",
            "Generate a random English musical instrument name that would be ungrammatical in most contexts."
        ]
        strategy = strategies[attempt % len(strategies)]
        if self.backend == "LOCAL":
            return self._generate_fallback_word_local(strategy)
        elif self.backend == "HF_API":
            return self._generate_fallback_word_hf_api(strategy)
        elif self.backend == "GROQ_API":
            return self._generate_fallback_word_groq_api(strategy)
        else:
            return None

    def _generate_fallback_word_local(self, strategy: str) -> str:
        try:
            system_msg = f"Generate ONE English word based on this instruction: {strategy}. Output only the word, no explanation."
            inputs = self.tokenizer.encode(system_msg, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=3,
                    temperature=1.2,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            text = generated_text[len(system_msg):].strip()
            words = re.findall(r"[A-Za-z'-]+", text)
            return words[0].lower() if words else None
        except Exception:
            return None

    def _generate_fallback_word_hf_api(self, strategy: str) -> str:
        try:
            resp = self.hf.chat_completion(
                messages=[
                    {"role": "system", "content": f"Generate ONE English word based on this instruction: {strategy}. Output only the word, no explanation."}
                ],
                max_tokens=10,
                temperature=1.2,
            )
            text = resp.choices[0].message["content"].strip()
            words = re.findall(r"[A-Za-z'-]+", text)
            return words[0].lower() if words else None
        except Exception:
            return None

    def _generate_fallback_word_groq_api(self, strategy: str) -> str:
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": f"Generate ONE English word based on this instruction: {strategy}. Output only the word, no explanation."}
                ],
                "max_tokens": 10,
                "temperature": 1.2,
                "stream": False
            }
            response = requests.post(self.groq_base_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            text = result["choices"][0]["message"]["content"].strip()
            words = re.findall(r"[A-Za-z'-]+", text)
            return words[0].lower() if words else None
        except Exception:
            return None

    def _generate_emergency_fallback_word(self) -> str:
        try:
            if self.backend == "LOCAL":
                return self._generate_fallback_word_local("Generate any random English word.")
            elif self.backend == "HF_API":
                return self._generate_fallback_word_hf_api("Generate any random English word.")
            elif self.backend == "GROQ_API":
                return self._generate_fallback_word_groq_api("Generate any random English word.")
        except Exception:
            pass
        return None

    # ================== eval model ==================

    def _is_good_word(self, w: str) -> bool:
        if not w: return False
        wl = w.lower()

    # shape restriction
        if not re.fullmatch(r"[a-z]{3,12}", wl):
            return False

        if wl in self.FORBIDDEN_WORDS:
            return False

    # repeat check
        if hasattr(self, "sentence_lexicon") and wl in self.sentence_lexicon:
            return False

    # lemma check
        lem = self._lemma(wl)
        if hasattr(self, "sentence_lemmas") and lem in self.sentence_lemmas:
            return False
        if hasattr(self, "used_lemmas_in_sentence") and lem in self.used_lemmas_in_sentence:
            return False

    # Ngram >0 ro avoid nonce
        try:
            if self.ngram_api.get_ngram_frequency(w) <= 0.0:
                return False
        except Exception:
            return False

    # modern word (not archaic/obsolete)
        wf = self.wordfreq.get(wl, 0.0) if hasattr(self, "wordfreq") else 1.0
        if wf < getattr(self, "min_wordfreq", 5e-6):
            return False

    # ban proper nouns
        try:
            if self._get_word_pos(w) == "PROPN":
                return False
        except Exception:
            pass

    
        if wl in self.used_words_in_sentence:
            return False
        if self.word_frequency_tracker.get(wl, 0) >= 3:
            return False

        return True

    def compute_surprisal(self, context: str, candidate: str) -> float:
        alpha = 1.0 / math.log(2)
        return alpha * (len(context) + len(candidate))

    def cosine_similarity(self, w1: str, w2: str) -> float:
        d1, d2 = self.nlp(w1), self.nlp(w2)
        if d1.has_vector and d2.has_vector:
            return d1.similarity(d2)
        return 0.0

    def _is_syntactically_valid(self, context: str, candidate: str) -> bool:
        sentence = (context + " " + candidate).strip()
        if not sentence:
            return False
        return not self.grammar_checker.has_grammar_error(sentence)

    # ================== tools==================

    def _get_word_pos(self, word: str) -> str:
        if not word or not word.strip():
            return "UNK"
        doc = self.nlp(word)
        return doc[0].pos_ if len(doc) else "UNK"

    def _lemma(self, w: str) -> str:
        if not w: return ""
        d = self.nlp(w)
        return d[0].lemma_.lower() if len(d) else w.lower()

    def _is_pos(self, word: str, pos: str) -> bool:
        if not word or not pos: return False
        d = self.nlp(word)
        return len(d) > 0 and d[0].pos_ == pos

    def _is_base_form(self, w: str, desired_pos: str | None) -> bool:
        """要求基本形：VERB=VB（bare），NOUN=单数，ADJ/ADV=原级；禁止 PROPN/PRON"""
        if not w: return False
        d = self.nlp(w)
        if not d: return False
        t = d[0]
        if t.pos_ in ("PROPN", "PRON"):  
            return False
        if desired_pos == "VERB":
            return t.pos_ == "VERB" and (t.tag_ == "VB" or "Inf" in t.morph.get("VerbForm"))
        if desired_pos == "NOUN":
            return t.pos_ == "NOUN" and (t.morph.get("Number") in (["Sing"], []))
        if desired_pos == "ADJ":
            return t.pos_ == "ADJ" and (t.morph.get("Degree") in (["Pos"], []))
        if desired_pos == "ADV":
            return t.pos_ == "ADV" and (t.morph.get("Degree") in (["Pos"], []))

        return t.pos_ not in ("ADP","AUX","CCONJ","SCONJ","PART","DET","PRON","PROPN")

    def _choose_desired_pos(self, target_pos: str) -> str:
        """
        Choose a desired POS that is different from the target POS.
        """
        import random
        if target_pos in ("NOUN", "PROPN"):
            return random.choice(["VERB", "ADJ"])
        if target_pos in ("VERB", "AUX"):
            return random.choice(["NOUN", "ADJ"])
        if target_pos in ("ADJ", "ADV"):
            return random.choice(["NOUN", "VERB"])
        return random.choice(["NOUN", "VERB"])

    def _create_first_pair(self, word: str) -> dict:
        return {"position": 0, "correct": word, "distractor": "x-x-x"}

    def _is_punct_only(self, token: str) -> bool:
        """ Check if the token consists solely of punctuation characters."""
        return bool(re.fullmatch(r"\W+", token))

    def reset_global_word_tracker(self):
        """ Clear the global word usage tracker."""
        self.word_frequency_tracker = {}


if __name__ == "__main__":
    generator = MazeGenerator()
    test_sentence = "Megan got angry with her boss and sued him for discrimination last month."
    result = generator.generate_full_maze(test_sentence)
    print("generating results:")
    for pair in generator.generate_full_maze_stream(test_sentence):
        print(f"[Position {pair['position']}] {pair['correct']} vs {pair['distractor']}")
