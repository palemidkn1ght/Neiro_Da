import os
import re
import torch
import torch.nn as nn
import numpy as np

# 1. Загрузка данных
def load_dataset():
    texts_file = "data/texts.txt"
    labels_file = "data/labels.txt"

    if not os.path.exists(texts_file) or not os.path.exists(labels_file):
        print("Не найдены файлы data/texts.txt или data/labels.txt")
        return [], []

    with open(texts_file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    with open(labels_file, "r", encoding="utf-8") as f:
        labels = [int(line.strip()) for line in f if line.strip()]

    if len(texts) != len(labels):
        print("Количество строк не совпадает!")
        return [], []

    print(f"Загружено {len(texts)} примеров.")
    return texts, labels



# 2. Предобработка
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^а-яёa-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_vocabulary(texts):
    vocab = {}
    for text in texts:
        for word in clean_text(text).split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


def vectorize(text, vocab):
    vec = np.zeros(len(vocab))
    for word in clean_text(text).split():
        if word in vocab:
            vec[vocab[word]] = 1
    return vec

# 3. Простая нейросеть
class AgeModel(nn.Module):
    def __init__(self, input_size):
        super(AgeModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)  # выход — одно число (рейтинг)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 4. Основной код
if __name__ == "__main__":
    print("Загружаем данные из файлов...")
    texts, labels = load_dataset()

    if not texts:
        exit()


    vocab = build_vocabulary(texts)
    print("Размер словаря:", len(vocab))

    # Преобразуем тексты в векторы
    X = np.array([vectorize(t, vocab) for t in texts], dtype=np.float32)
    Y = np.array(labels, dtype=np.float32).reshape(-1, 1)

    # Переводим в torch.Tensor
    X_train = torch.tensor(X)
    Y_train = torch.tensor(Y)

    # Создаём модель
    model = AgeModel(input_size=len(vocab))
    loss_fn = nn.MSELoss()  # среднеквадратичная ошибка
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Обучаем
    print("Обучение модели:")
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(X_train)
        loss = loss_fn(output, Y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Эпоха {epoch+1}/100 — Потеря: {loss.item():.4f}")

    # Проверяем предсказания
    print("Проверка результатов:")
    with torch.no_grad():
        preds = model(X_train).round().numpy()
        for text, pred, true in zip(texts, preds, labels):
            print(f"{text:30s} | Предсказано: {int(pred[0])}+ | Истинное: {true}+")

#да простят меня все боги кодинга за этот вайбкод
