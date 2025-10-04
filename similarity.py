# similarity
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

# 初始化模型和分词器
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

def get_similarity(word1: str, word2: str) -> float:
    """计算两个单词的语义相似度（Cosine Similarity）"""
    with torch.no_grad():
        inputs1 = tokenizer(word1, return_tensors="pt", truncation=True, max_length=8)
        inputs2 = tokenizer(word2, return_tensors="pt", truncation=True, max_length=8)

        emb1 = model(**inputs1).last_hidden_state[:, 0, :]
        emb2 = model(**inputs2).last_hidden_state[:, 0, :]

        sim = cosine_similarity(emb1.numpy(), emb2.numpy())[0][0]
        return round(float(sim), 4)