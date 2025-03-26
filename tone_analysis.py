from typing import Literal
import pandas as pd
from transformers import pipeline
import logging
from tqdm.auto import tqdm
import torch
import matplotlib.pyplot as plt
from IPython.display import display

logging.basicConfig(level=logging.INFO)


class ToneAnalysis:
    logger = logging.getLogger(__name__)
    sentiment_analyzer = pipeline(
        task="text-classification",
        model="seara/rubert-tiny2-russian-sentiment",
        framework="pt",
        device=torch.device("cuda:0")
        if torch.cuda.is_available()
        else torch.device("cpu"),
    )

    def __init__(self, filename: str, teacher_name: str | None = None):
        if filename.endswith(".xlsx"):
            self.data = pd.read_excel(filename)
        elif filename.endswith(".csv"):
            self.data = pd.read_csv(filename)
        else:
            raise ValueError("Неверный формат файла")

        self.teacher_name = teacher_name

    def tone_analysis(self):
        self.analyze_sentiment()
        self.data.to_csv("data_with_sentiment.csv", index=False)

        # Распределение тональностей
        print("\nРаспределение тональности:")
        print(self.data["sentiment"].value_counts(normalize=True))

        # Пример оценки
        for label in ["positive", "negative", "neutral"]:
            examples = self.data[self.data["sentiment"] == label].head(2)
            print(f"\nПримеры ({label}):")
            for _, row in examples.iterrows():
                print(f"Текст: {row['Комментарий студента'][:100]}...")
                print(f"Тональность: {row['sentiment']}\n{'-' * 50}")

        print("\nАнализ всех преподавателей:")
        self.analyze_all_teachers_feedback()

        if self.teacher_name:
            self.analyze_single_teacher_feedback(self.teacher_name)
            return

        print("\nАнализ позитивных отзывов:")
        self.analyze_feedback("positive")

        print("\nАнализ негативных отзывов:")
        self.analyze_feedback("negative")

    def analyze_all_teachers_feedback(self):
        """
        Анализ отзывов для всех преподавателей.
        """
        self.all_teachers_stats = (
            self.data.groupby("ФИО преподавателя")["sentiment"]
            .value_counts()
            .unstack()
            .fillna(0)
        )
        self.all_teachers_stats["Всего отзывов"] = self.all_teachers_stats.sum(axis=1)
        display(
            self.all_teachers_stats.sort_values("Всего отзывов", ascending=False).head(
                10
            )
        )

    def analyze_single_teacher_feedback(self, teacher_name: str):
        """
        Анализ отзывов для одного преподавателя.
        """
        teacher_df = self.data[self.data["ФИО преподавателя"] == teacher_name]

        if teacher_df.empty:
            print(f"Преподаватель '{teacher_name}' не найден в данных")
            return

        sentiment_stats = (
            teacher_df["sentiment"]
            .value_counts()
            .reindex(["positive", "neutral", "negative"], fill_value=0)
        )

        emotion_stats = teacher_df["sentiment"].value_counts()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        sentiment_stats.plot(kind="bar", ax=ax1, color=["green", "blue", "red"])
        ax1.set_title(f"Распределение тональности для {teacher_name}")
        ax1.set_xlabel("Тональность")
        ax1.set_ylabel("Количество отзывов")

        emotion_stats.plot(kind="pie", ax=ax2, autopct="%1.1f%%")
        ax2.set_title(f"Распределение эмоций для {teacher_name}")
        ax2.set_ylabel("")

        plt.tight_layout()
        plt.show()

        report = f"""
        Анализ отзывов для преподавателя: {teacher_name}
        Всего отзывов: {len(teacher_df)}
        
        Тональность:
        - Позитивные: {sentiment_stats.get("positive", 0)} 
        - Нейтральные: {sentiment_stats.get("neutral", 0)}
        - Негативные: {sentiment_stats.get("negative", 0)}
        
        Основные эмоции:
        {emotion_stats.head(3).to_string()}
        """
        print(report)

        print("\nПримеры отзывов:")
        display(
            teacher_df[["Комментарий студента", "sentiment"]].sample(3, random_state=42)
        )

    def analyze_feedback(
        self, sentiment: Literal["positive", "negative"], min_reviews=5
    ):
        teacher_stats = (
            self.data.groupby("ФИО преподавателя")
            .agg(
                total_reviews=("sentiment", "count"),
                sentiment_reviews=("sentiment", lambda x: (x == sentiment).sum()),
            )
            .reset_index()
        )

        teacher_stats = teacher_stats[teacher_stats["total_reviews"] >= min_reviews]
        teacher_stats[f"{sentiment}_percent"] = (
            teacher_stats["sentiment_reviews"] / teacher_stats["total_reviews"]
        ) * 100

        # Топ-10
        top = teacher_stats.sort_values("sentiment_reviews", ascending=False).head(10)

        # визуализация
        plt.figure(figsize=(12, 8))
        plt.barh(
            top["ФИО преподавателя"],
            top["sentiment_reviews"],
            color="red",
        )
        plt.title(f"Топ-10 преподавателей по количеству {sentiment} отзывов")
        plt.xlabel(f"Количество {sentiment} отзывов")
        plt.ylabel("Преподаватель")
        plt.gca().invert_yaxis()
        plt.show()

        avg = (
            self.data["sentiment"].value_counts(normalize=True).get(sentiment, 0) * 100
        )
        print(f"Средний процент {sentiment} отзывов по всем преподавателям: {avg:.1f}%")

        # Детальная таблица
        teacher_stats[f"{sentiment}_percent"] = teacher_stats[
            f"{sentiment}_percent"
        ].round(1)
        teacher_stats = teacher_stats.sort_values(
            f"{sentiment}_percent", ascending=False
        )

        print(f"\nДетальная статистика по {sentiment} отзывам:")
        display(
            teacher_stats[
                [
                    "ФИО преподавателя",
                    "total_reviews",
                    "sentiment_reviews",
                    f"{sentiment}_percent",
                ]
            ]
        )

        self.all_teachers_stats[f"Доля {sentiment} (%)"] = (
            self.all_teachers_stats[f"{sentiment}"]
            / self.all_teachers_stats["Всего отзывов"]
            * 100
        ).round(1)
        display(
            self.all_teachers_stats.sort_values(f"{sentiment}", ascending=False).head(
                10
            )
        )

    def analyze_sentiment(self, batch_size: int = 100):
        """
        Анализ тональности комментариев студентов. Производится батчами, результат сохраняется в self.data.
        """
        self.data["sentiment"] = "neutral"
        total_rows = len(self.data)

        for i in tqdm(range(0, total_rows, batch_size), desc="Обработка тональности"):
            batch = self.data["Комментарий студента"].iloc[i : i + batch_size].tolist()
            batch_results = self._analyze_sentiment_batch(batch)
            self.data.loc[i : i + batch_size - 1, "sentiment"] = batch_results

    def _analyze_sentiment_batch(self, batch: list[str]) -> list[str]:
        results = []
        for text in batch:
            try:
                # Проверка и очистка текста
                if pd.isna(text) or not isinstance(text, str):
                    results.append("neutral")
                    continue

                cleaned_text = text.strip()
                if len(cleaned_text) == 0:
                    results.append("neutral")
                    continue

                result = self.sentiment_analyzer(
                    cleaned_text, truncation=True, max_length=512
                )[0]
                results.append(result["label"])
            except Exception as e:
                self.logger.error(
                    f"Ошибка при обработке текста: {text[:50]}... | Ошибка: {str(e)}"
                )
                results.append("neutral")
        return results
