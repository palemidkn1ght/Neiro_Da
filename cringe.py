import os
import re
import torch
import torch.nn as nn
import numpy as np


# ===== ЗАГРУЗКА ДАННЫХ =====
def load_triggers():
    """Загрузка триггеров из файла"""
    triggers = {}

    with open("data/triggers.txt", "r", encoding="utf-8") as f:
        current_category = None
        for line in f:
            line = line.strip()
            if line and not line[0].islower():  # Категория - с большой буквы
                current_category = line.lower()
                triggers[current_category] = []
            elif line and current_category:
                triggers[current_category].append(line.lower())

    print(f"Загружено категорий: {len(triggers)}")
    return triggers


def load_training_data():
    """Загрузка данных для обучения"""
    texts = []
    labels = []

    # Загружаем тексты
    with open("data/texts.txt", "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    # Загружаем метки
    with open("data/labels.txt", "r", encoding="utf-8") as f:
        labels = [int(line.strip()) for line in f if line.strip()]

    print(f"Загружено {len(texts)} примеров для обучения")
    return texts, labels


# ===== ПРЕДОБРАБОТКА =====
def clean_text(text):
    """Очистка текста"""
    text = text.lower()
    text = re.sub(r"[^а-яёa-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def text_to_features(text, triggers):
    """Преобразование текста в числовые признаки"""
    cleaned = clean_text(text)
    words = cleaned.split()

    features = []

    # Для каждой категории добавляем два признака:
    for category, trigger_words in triggers.items():
        # 1. Количество триггеров этой категории
        count = sum(1 for word in words if word in trigger_words)
        features.append(count)

        # 2. Наличие хотя бы одного триггера (0 или 1)
        features.append(1 if count > 0 else 0)

    return features


# ===== МОДЕЛЬ =====
class RatingModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 5)  # 5 классов: 0+, 6+, 12+, 16+, 18+

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ===== ОБУЧЕНИЕ =====
def train_model():
    """Обучение модели"""
    print("=== ОБУЧЕНИЕ МОДЕЛИ ===")

    # Загружаем данные
    triggers = load_triggers()
    texts, labels = load_training_data()

    if not texts:
        print("Нет данных для обучения!")
        return None, triggers

    # Преобразуем тексты в признаки
    X = []
    y = []

    for text, label in zip(texts, labels):
        features = text_to_features(text, triggers)
        X.append(features)

        # Преобразуем рейтинг в класс (0->0, 6->1, 12->2, 16->3, 18->4)
        rating_to_class = {0: 0, 6: 1, 12: 2, 16: 3, 18: 4}
        y.append(rating_to_class[label])

    # Преобразуем в тензоры PyTorch
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Создаем модель
    input_size = len(X[0])
    model = RatingModel(input_size)

    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Процесс обучения
    print("Начинаем обучение...")
    for epoch in range(100):
        # Прямой проход
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)

        # Обратный проход
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Эпоха {epoch + 1}, Потери: {loss.item():.4f}")

    # Сохраняем модель
    torch.save(model.state_dict(), "model.pth")
    print("Модель сохранена как 'model.pth'")

    return model, triggers


# ===== ПРЕДСКАЗАНИЕ =====
def predict_rating(model, triggers, text):
    """Предсказание рейтинга для текста"""
    features = text_to_features(text, triggers)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(features_tensor)
        predicted_class = torch.argmax(output, 1).item()

    # Преобразуем класс обратно в рейтинг
    class_to_rating = {0: 0, 1: 6, 2: 12, 3: 16, 4: 18}
    return class_to_rating[predicted_class]


def analyze_triggers(text, triggers):
    """Анализ триггеров в тексте"""
    cleaned = clean_text(text)
    words = cleaned.split()

    found = {}
    for category, trigger_words in triggers.items():
        found_words = [word for word in trigger_words if word in words]
        if found_words:
            found[category] = found_words

    return found


# ===== ОСНОВНАЯ ПРОГРАММА =====
def main():
    # Обучаем или загружаем модель
    if os.path.exists("model.pth"):
        print("Загружаем существующую модель...")
        triggers = load_triggers()
        input_size = len(text_to_features("тест", triggers))
        model = RatingModel(input_size)
        model.load_state_dict(torch.load("model.pth"))
    else:
        print("Создаем новую модель...")
        model, triggers = train_model()
        if model is None:
            return

    # Тестируем на новых текстах
    print("\n=== ТЕСТИРОВАНИЕ ===")

    with open("data/test_texts.txt", "r", encoding="utf-8") as f:
        test_texts = [line.strip() for line in f if line.strip()]

    for i, text in enumerate(test_texts, 1):
        # Предсказываем рейтинг
        rating = predict_rating(model, triggers, text)

        # Анализируем триггеры
        found_triggers = analyze_triggers(text, triggers)

        print(f"{i}. Текст: {text}")
        print(f"   Рейтинг: {rating}+")

        if found_triggers:
            print("   Найденные триггеры:")
            for category, words in found_triggers.items():
                print(f"     - {category}: {', '.join(words)}")
        print()


if __name__ == "__main__":
    main()