import os
import re
import torch
import torch.nn as nn
import numpy as np


# ===== ЗАГРУЗКА ТРИГГЕРОВ ИЗ ФАЙЛА =====
def load_triggers():
    """Загружает триггеры из файла triggers.txt"""
    triggers_file = "data/triggers.txt"
    triggers = {}

    if not os.path.exists(triggers_file):
        print(f"Файл {triggers_file} не найден")
        return triggers

    with open(triggers_file, "r", encoding="utf-8") as f:
        current_category = None

        for line in f:
            line = line.strip()
            if not line:
                continue

            # Если строка начинается с категории
            if line.endswith(':'):
                current_category = line[:-1]
                triggers[current_category] = []
            elif current_category and line:
                # Добавляем слова в текущую категорию
                triggers[current_category].append(line)

    print(f"Загружено категорий триггеров: {len(triggers)}")
    return triggers


# ===== ЗАГРУЗКА ДАННЫХ =====
def load_dataset():
    texts_file = "data/texts.txt"
    labels_file = "data/labels.txt"

    if not os.path.exists(texts_file) or not os.path.exists(labels_file):
        print("Не найдены файлы data/texts.txt или data/labels.txt")
        return [], []

    with open(texts_file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    with open(labels_file, "r", encoding="utf-8") as f:
        rating_map = {'0': 0, '6': 6, '12': 12, '16': 16, '18': 18}
        labels = []
        for line in f:
            line = line.strip()
            if line:
                labels.append(rating_map.get(line, 0))

    if len(texts) != len(labels):
        print("Количество строк не совпадает!")
        return [], []

    print(f"Загружено {len(texts)} примеров.")
    return texts, labels


def load_test_texts():
    """Загружает тестовые тексты из отдельного файла"""
    test_file = "data/test_texts.txt"

    if not os.path.exists(test_file):
        print(f"Не найден файл {test_file}")
        return []

    with open(test_file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    print(f"Загружено {len(texts)} тестовых текстов.")
    return texts


# ===== АНАЛИЗ ТРИГГЕРОВ =====
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^а-яёa-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def analyze_triggers(text, triggers):
    """Анализирует текст и возвращает детальную информацию о триггерах"""
    cleaned_text = clean_text(text)
    words = cleaned_text.split()

    # Считаем триггеры по категориям
    trigger_counts = {category: 0 for category in triggers}
    found_triggers = {category: [] for category in triggers}

    for word in words:
        for category, category_triggers in triggers.items():
            if word in category_triggers:
                trigger_counts[category] += 1
                if word not in found_triggers[category]:
                    found_triggers[category].append(word)

    # Создаем вектор признаков
    features = []
    for category in triggers:
        features.append(trigger_counts[category])
        features.append(1.0 if trigger_counts[category] > 0 else 0.0)

    features.append(len(words))

    return np.array(features, dtype=np.float32), trigger_counts, found_triggers


def calculate_rating_from_triggers(trigger_counts, triggers):
    """Вычисляет рейтинг на основе триггеров по четким правилам"""
    # Правила определения рейтинга
    if any(trigger_counts.get(category, 0) > 0 for category in ['adult_content', 'explicit']):
        return 18
    elif trigger_counts.get('drugs', 0) >= 2 or trigger_counts.get('violence', 0) >= 3:
        return 18
    elif trigger_counts.get('drugs', 0) >= 1 or trigger_counts.get('violence', 0) >= 2:
        return 16
    elif trigger_counts.get('violence', 0) >= 1 or trigger_counts.get('fear', 0) >= 2:
        return 12
    elif trigger_counts.get('language', 0) >= 1:
        return 6
    else:
        return 0


# ===== ОСНОВНОЙ КОД =====
if __name__ == "__main__":
    print("=== СИСТЕМА ОПРЕДЕЛЕНИЯ ВОЗРАСТНЫХ РЕЙТИНГОВ ===")

    # Загружаем триггеры
    print("Загружаем триггеры...")
    triggers = load_triggers()

    if not triggers:
        print("Триггеры не загружены!")
        exit()

    # Загружаем тестовые тексты
    print("Загружаем тестовые тексты...")
    test_texts = load_test_texts()

    if not test_texts:
        print("Тестовые тексты не найдены!")
        exit()

    # Анализируем тестовые тексты
    print("\nРЕЗУЛЬТАТЫ АНАЛИЗА:")
    print("=" * 50)

    for i, text in enumerate(test_texts, 1):
        # Получаем признаки и анализ триггеров
        features, trigger_counts, found_triggers = analyze_triggers(text, triggers)

        # Определяем рейтинг по правилам триггеров
        rating = calculate_rating_from_triggers(trigger_counts, triggers)

        # Выводим результаты
        print(f"{i}. Текст: {text}")
        print(f"   Рейтинг: {rating}+")

        # Показываем найденные триггеры, которые повлияли на рейтинг
        affecting_categories = []
        for category, count in trigger_counts.items():
            if count > 0:
                affecting_categories.append(category)

        if affecting_categories:
            print(f"   Триггеры: {', '.join(affecting_categories)}")

            # Детализация по конкретным словам
            for category in affecting_categories:
                if found_triggers[category]:
                    triggers_list = ', '.join(found_triggers[category])
                    print(f"      {category}: {triggers_list}")
        else:
            print("   Триггеры: не найдены")

        print()