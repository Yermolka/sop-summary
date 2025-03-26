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

    def __init__(self, filename: str):
        if filename.endswith(".xlsx"):
            self.data = pd.read_excel(filename)
        elif filename.endswith(".csv"):
            self.data = pd.read_csv(filename)
        else:
            raise ValueError("Неверный формат файла")

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

        print("\nАнализ позитивных отзывов:")
        self.analyze_positive_feedback()

        print("\nАнализ негативных отзывов:")
        self.analyze_negative_feedback()

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
            teacher_df[["Комментарий студента", "sentiment", "emotion"]].sample(
                3, random_state=42
            )
        )

    def analyze_negative_feedback(self, min_reviews=5):
        """
        Анализ негативных отзывов для всех преподавателей.
        """
        teacher_stats = (
            self.data.groupby("ФИО преподавателя")
            .agg(
                total_reviews=("sentiment", "count"),
                negative_reviews=("sentiment", lambda x: (x == "negative").sum()),
            )
            .reset_index()
        )

        teacher_stats = teacher_stats[teacher_stats["total_reviews"] >= min_reviews]
        teacher_stats["negative_percent"] = (
            teacher_stats["negative_reviews"] / teacher_stats["total_reviews"]
        ) * 100

        # Топ-10
        top_negative = teacher_stats.sort_values(
            "negative_reviews", ascending=False
        ).head(10)

        # визуализация
        plt.figure(figsize=(12, 8))
        plt.barh(
            top_negative["ФИО преподавателя"],
            top_negative["negative_reviews"],
            color="red",
        )
        plt.title("Топ-10 преподавателей по количеству негативных отзывов")
        plt.xlabel("Количество негативных отзывов")
        plt.ylabel("Преподаватель")
        plt.gca().invert_yaxis()
        plt.show()

        avg_negative = (
            self.data["sentiment"].value_counts(normalize=True).get("negative", 0) * 100
        )
        print(
            f"Средний процент негативных отзывов по всем преподавателям: {avg_negative:.1f}%"
        )

        # Детальная таблица
        teacher_stats["negative_percent"] = teacher_stats["negative_percent"].round(1)
        teacher_stats = teacher_stats.sort_values("negative_percent", ascending=False)

        print("\nДетальная статистика по негативным отзывам:")
        display(
            teacher_stats[
                [
                    "ФИО преподавателя",
                    "total_reviews",
                    "negative_reviews",
                    "negative_percent",
                ]
            ]
        )

        self.all_teachers_stats["Доля негатива (%)"] = (
            self.all_teachers_stats["negative"]
            / self.all_teachers_stats["Всего отзывов"]
            * 100
        ).round(1)
        display(
            self.all_teachers_stats.sort_values("negative", ascending=False).head(10)
        )

    def analyze_positive_feedback(self, min_reviews=5):
        """
        Анализ позитивных отзывов для всех преподавателей.
        """
        teacher_stats = (
            self.data.groupby("ФИО преподавателя")
            .agg(
                total_reviews=("sentiment", "count"),
                positive_reviews=("sentiment", lambda x: (x == "positive").sum()),
            )
            .reset_index()
        )

        teacher_stats = teacher_stats[teacher_stats["total_reviews"] >= min_reviews]
        teacher_stats["positive_percent"] = (
            teacher_stats["positive_reviews"] / teacher_stats["total_reviews"]
        ) * 100

        top_positive = teacher_stats.sort_values(
            "positive_reviews", ascending=False
        ).head(10)

        plt.figure(figsize=(12, 8))
        plt.barh(
            top_positive["ФИО преподавателя"],
            top_positive["positive_reviews"],
            color="green",
        )
        plt.title("Топ-10 преподавателей по количеству позитивных отзывов")
        plt.xlabel("Количество позитивных отзывов")
        plt.ylabel("Преподаватель")
        plt.gca().invert_yaxis()
        plt.show()

        avg_positive = (
            self.data["sentiment"].value_counts(normalize=True).get("positive", 0) * 100
        )
        print(
            f"Средний процент позитивных отзывов по всем преподавателям: {avg_positive:.1f}%"
        )

        teacher_stats["positive_percent"] = teacher_stats["positive_percent"].round(1)
        teacher_stats = teacher_stats.sort_values("positive_percent", ascending=False)

        print("\nДетальная статистика по позитивным отзывам:")
        display(
            teacher_stats[
                [
                    "ФИО преподавателя",
                    "total_reviews",
                    "positive_reviews",
                    "positive_percent",
                ]
            ]
        )

        print("\nОбщая статистика с позитивными отзывами:")
        self.all_teachers_stats["Доля позитива (%)"] = (
            self.all_teachers_stats["positive"]
            / self.all_teachers_stats["Всего отзывов"]
            * 100
        ).round(1)
        display(
            self.all_teachers_stats.sort_values("positive", ascending=False).head(10)
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
