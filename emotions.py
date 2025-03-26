import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class EmotionAnalysis:
    EMOTION_LABELS = [
        "neutral",  # 0
        "joy",  # 1
        "sadness",  # 2
        "surprise",  # 3
        "fear",  # 4
        "anger",  # 5
    ]

    model_name = "Djacon/rubert-tiny2-russian-emotion-detection"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    def __init__(self, filename: str):
        self.data = pd.read_csv(filename)

    def analyze(self):
        self.average_sentiment()
        self.emotional_analysis()

    def average_sentiment(self):
        """
        Выдает таблицу(average_sentiment_full_metadata)со средним арифметическим по удовлетворенности студентов курсом
        """

        columns_to_keep = [
            "Семестр",
            "ОП",
            "Курс",
            "Дисциплина",
            "Ключ дисциплины",
            "sentiment",
        ]
        data = self.data[columns_to_keep].copy()

        # Преобразование sentiment в числовые значения, "коэффицент удовлетворенности курсом" - среднее арифметическое по всем отзывам по данному курсу. принимает значения от -1(все отзывы негативные) до 1(все отзывы позитивные)
        sentiment_mapping = {"positive": 1, "neutral": 0, "negative": -1}
        data["sentiment_numeric"] = data["sentiment"].map(sentiment_mapping)

        # Группировка с сохранением всех метаданных (правильный синтаксис)
        result = (
            data.groupby(["Семестр", "ОП", "Курс", "Дисциплина", "Ключ дисциплины"])[
                "sentiment_numeric"
            ]
            .mean()
            .reset_index()
        )
        result.columns = [
            "Семестр",
            "ОП",
            "Курс",
            "Название дисциплины",
            "Ключ дисциплины",
            "Средний sentiment",
        ]

        # Сортировка по курсу и названию дисциплины
        result.sort_values(["Курс", "Название дисциплины"], inplace=True)

        # Сохранение результатов
        result.to_csv(
            "average_sentiment_full_metadata.csv", index=False, encoding="utf-8-sig"
        )

        # Вывод результатов
        print("Результаты сохранены в файл 'average_sentiment_full_metadata.csv'")
        print(result.head())

    def emotional_analysis(self):
        data = self.data.copy()
        # Анализируем комментарии с прогресс-баром
        print("Начинаем анализ эмоций в комментариях...")
        tqdm.pandas(desc="Обработка комментариев")
        data["emotion"] = data["Комментарий студента"].progress_apply(
            self._predict_emotions
        )

        # Сохраняем результаты
        data.to_csv("comments_with_emotions.csv", index=False, encoding="utf-8-sig")

        print("\nАнализ завершен. Результаты сохранены в 'comments_with_emotions.csv'")
        print("\nРаспределение эмоций:")
        print(data["emotion"].value_counts())

        print("\nПример результатов:")
        print(data[["Комментарий студента", "emotion"]].head(10))

    def _predict_emotions(self, text: str) -> str:
        try:
            if pd.isna(text) or str(text).strip() == "":
                return "no_text"

            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            emotion_idx = torch.argmax(probabilities).item()

            # Проверка, что индекс в допустимых пределах
            if 0 <= emotion_idx < len(self.EMOTION_LABELS):
                return self.EMOTION_LABELS[emotion_idx]
            else:
                return f"unknown_{emotion_idx}"

        except Exception as e:
            return f"error_{str(e)}"
