import numpy as np
import re
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import Optional, List, Dict
from dataclasses import dataclass

with open('config.json', 'r') as f:
    config = json.load(f)

# Параметры из конфига
qdrant_url = config['qdrant']['url']
qdrant_api_key = config['qdrant']['api_key']
qd_collection_name = config['qdrant']['collection_name']
embedding_model = config['model']['embedding_model']
top_k_results = int(config['model']['top_k_results'])

# Инициализация моделей и клиента
model = SentenceTransformer(embedding_model)
client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

# Выбор и настройка модели
use_local_llm = True
llm_model_name = "Qwen/Qwen2.5-1.5B-Instruct"

rag_promt_template = """
Ты — AI-ассистент университета ИТМО. Отвечаешь на вопросы абитуриентов о магистерских программах "Искусственный Интеллект" и "Управление ИИ-продуктами/AI Product".
Используй ТОЛЬКО предоставленный контекст. Если ответа нет в контексте, говори "У меня нет информации по этому вопросу".

Правила ответа:
1. Будь точным и используй только факты из контекста
2. Сохраняй официально-дружелюбный тон
3. Форматируй ответ: краткое введение → основные пункты → заключение
4. Для перечислений используй маркированные списки

Контекст:
{context}

Вопрос: {question}
Ответ:
"""

if use_local_llm:
    _tok = AutoTokenizer.from_pretrained(llm_model_name)
    _llm = AutoModelForCausalLM.from_pretrained(llm_model_name, torch_dtype=torch.float32)
    _llm.eval()

# Поиск данных
def retrieve_for_rag(query: str, top_k: int = top_k_results):
    query_embedding = model.encode(f"query: {query}", normalize_embeddings=True).tolist()
    results = client.search(
        collection_name=qd_collection_name,
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True,
        with_vectors=False
    )

    return [
        {
            "text": hit.payload["text"],
            "metadata": {
                "program_name": hit.payload.get("program_name", "Искусственный Интеллект"),
                "section": hit.payload.get("section", "")
            }
        }
        for hit in results
        if "text" in hit.payload  # Фильтруем только чанки с текстом
    ]

# Генерация ответа
def generate_answer(query: str, contexts: list) -> str:
    if not contexts:
        return "У меня нет информации по этому вопросу."

    if use_local_llm:
        context_str = "\n\n".join(
            f"{c['text']}\n[Программа: {c['metadata']['program_name']}]"
            for c in contexts
        )

        prompt = rag_promt_template.format(
            context=context_str,
            question=query
        )

        inputs = _tok(prompt, return_tensors="pt")
        out = _llm.generate(**inputs, max_new_tokens=500, do_sample=False)
        return _tok.decode(out[0], skip_special_tokens=True).split("Ответ:")[-1].strip()

    return contexts[0]["text"]

# Конечная функция для связи с ботом
def ask_question(query: str):
    contexts = retrieve_for_rag(query)
    return generate_answer(query, contexts)