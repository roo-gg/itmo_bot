import json
import torch
import numpy as np
import re
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

with open('config.json', 'r') as f:
    config = json.load(f)

with open('ai_product.json', 'r') as f:
    ai_product_data = json.load(f)

with open('ai.json', 'r') as f:
    ai_data = json.load(f)

# Параметры из конфига
qdrant_url = config['qdrant']['url']
qdrant_api_key = config['qdrant']['api_key']
qd_collection_name = config['qdrant']['collection_name']
embedding_model = config['model']['embedding_model']
top_k_results = int(config['model']['top_k_results'])

# Инициализация моделей и клиента
model = SentenceTransformer(embedding_model)
client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

# Разбиваем текст на предложения
def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

# Генерация эмбеддинга для текста
def embed_chunk(text):
    return model.encode(f"passage: {text}", normalize_embeddings=True)

# Обработка и разбиение данных программы
def chunk_program_info(obj, path=""):
    result = []
    stack = [(obj, path)]

    while stack:
        current, current_path = stack.pop()

        if isinstance(current, str):
            if current.startswith(('http://', 'https://')):
                continue

            for sentence in split_into_sentences(current):
                combined_text = f"{current_path}: {sentence}" if current_path else sentence
                result.append({
                    "program_name": "Искусственный Интеллект",
                    "section": current_path,
                    "text": combined_text
                })

        elif isinstance(current, list):
            for idx, item in enumerate(reversed(current)):
                next_path = f"{current_path}[{idx}]" if current_path else f"[{idx}]"
                stack.append((item, next_path))

        elif isinstance(current, dict):
            for key, value in reversed(current.items()):
                next_path = f"{current_path}.{key}" if current_path else key
                stack.append((value, next_path))

    return result

ai_chunks = chunk_program_info(ai_data, 'ai')
aip_chunks = chunk_program_info(ai_product_data, 'ai_product')

# Добавляет эмбеддинг для всего содержимого чанка
def add_embeddings_to_chunks(chunks, embed_func):
    for chunk in chunks:
        # Собираем полное содержимое для эмбеддинга
        full_content = f"""
        Программа: {chunk['program_name']}
        Раздел: {chunk['section']}
        Текст: {chunk['text']}
        """
        # Генерируем и сохраняем эмбеддинг
        chunk["embedding"] = embed_func(full_content).tolist()
    return chunks

ai_chunks = add_embeddings_to_chunks(ai_chunks, embed_chunk)
aip_chunks = add_embeddings_to_chunks(aip_chunks, embed_chunk)

# Пересоздание коллекции в Qdrant

client.recreate_collection(collection_name = qd_collection_name,
                          vectors_config = models.VectorParams(
                                                            size = 1024,
                                                            distance = models.Distance.COSINE
                                                            )
                          )

# Преобразует чанки в точки для Qdrant
def chunks_to_points(chunks, points=None):
    if points is None:
        points = []

    start_idx = len(points)

    for i, chunk in enumerate(chunks, start=start_idx):
        vector = chunk["embedding"]

        # Проверка что вектор - список и имеет правильную размерность
        if not isinstance(vector, list) or len(vector) != 1024:
            print(f"Пропуск чанка {i}: неверная размерность вектора")
            continue

        points.append(
            models.PointStruct(
                id=i,
                vector=vector,
                payload={
                    "text": chunk["text"],
                    "program_name": chunk["program_name"],
                    "section": chunk["section"]
                }
            )
        )

    return points

points = chunks_to_points(ai_chunks)
points = chunks_to_points(aip_chunks, points)

#Добавление точек в Qdrant
operation_info = client.upsert(
                              collection_name = qd_collection_name,
                              points = points
                              )