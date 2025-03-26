import pandas as pd
import pymorphy3
import re
import nltk
from nltk.corpus import stopwords
from spellchecker import SpellChecker
from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast
import torch
from joblib import Parallel, delayed
import multiprocessing

nltk.download("stopwords")


class Preprocess:
    """
    Класс для предобработки данных. Для запуска требуется только путь к файлу с данными.
    """

    spell = SpellChecker(language="ru")
    morph = pymorphy3.MorphAnalyzer()

    stopwords_set = set(stopwords.words("russian"))

    # Инициализация модели
    MODEL_NAME = "UrukHan/t5-russian-spell"
    MAX_INPUT = 256
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def __init__(self, filename: str):
        self.data = pd.read_excel(filename)

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.model = self.model.to(self.device)
        else:
            self.device = torch.device("cpu")

    def preprocess(self):
        self.text_preprocess()
        self.lemmatize()
        self.remove_stopwords()

        self.data.to_excel("preprocessed.xlsx", index=False)
        self.data.to_csv("preprocessed.csv", index=False)

        # Применяем модель на небольшой выборке
        data1 = self.data.sample(n=100).copy()
        data1["Исправленный комментарий"] = data1["Комментарий студента"].apply(
            self.spell_ai
        )

        data1.to_excel("preprocessed_part_corrected.xlsx", index=False)
        data1.to_csv("preprocessed_part_corrected.csv", index=False)

    def text_preprocess(self) -> str:
        """
        Принимает на вход текст, возвращает текст в нижнем регистре без знаков
        препинания и без лишних пробелов.
        """

        def _text_preprocess(text: str) -> str:
            res = text.lower()
            res = res.replace("\t", " ").replace("\n", " ").replace("\r", " ")
            res = re.sub(r"[^\w\s]", " ", res)
            res = re.sub(r"\s+", " ", res)
            return res.strip()

        results = Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(_text_preprocess)(text)
            for text in self.data["Комментарий студента"]
        )
        self.data["text"] = results

    def correct_typos(self, text: str) -> str:
        """
        РАБОТАЕТ МЕДЛЕННО
        Принимает на вход текст в нижнем регистре без знаков препинания, возвращает
        этот текст с исправлеными ошибками (работает неидеально).
        """

        words = text.split(" ")

        misspelled = self.spell.unknown(words)

        corrected_words = ""

        for word in words:
            corrected_word = self.spell.correction(word)

            if word in misspelled and corrected_word:
                corrected_words += corrected_word + " "
            else:
                corrected_words += word + " "
        return corrected_words.strip()

    def remove_stopwords(self) -> str:
        """
        Принимает на вход текст в нижнем регистре без знаков препинания, возвращает
        текст с удаленными стоп-словами.
        """
        stopwords_set = self.stopwords_set

        def _remove_stopwords(l: str, r: str) -> tuple[str, str]:
            l_words = l.split(" ")
            r_words = r.split(" ")
            return (
                " ".join([word for word in l_words if word not in stopwords_set]),
                " ".join([word for word in r_words if word not in stopwords_set]),
            )

        results = Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(_remove_stopwords)(l, r)
            for l, r in zip(self.data["text"], self.data["lemmatized_text"])
        )
        self.data["text_without_stopwords"] = [l for l, _ in results]
        self.data["lemmatized_text_without_stopwords"] = [r for _, r in results]

    def lemmatize(self) -> str:
        """
        Принимает на вход текст после препроцессинга, приводит все слова в нем к начальным формам.
        """

        def _lemmatize(text: str) -> str:
            text = text.split(" ")
            res = ""
            for word in text:
                p = self.morph.parse(word)[0]
                res += p.normal_form + " "
            return res.strip()

        # Можно поставить True, но в случае нехватки ресурсов процесс будет убиваться
        fast: bool = False

        if fast:
            results = Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(_lemmatize)(text) for text in self.data["text"]
            )
            self.data["lemmatized_text"] = results
        else:
            self.data["lemmatized_text"] = self.data["text"].apply(_lemmatize)

    def clean_text(self, text: str) -> str:
        """
        РАБОТАЕТ МЕДЛЕННО ИЗ-ЗА ИСПРАВЛЕНИЯ ОШИБОК
        Принимает на вход текст в нижнем региcтре без знаков препинания, возвращает
        текст с исправленными ошибками (с переменным успехом), без стоп-слов и
        лематизированный.
        """
        words = text.split(" ")

        misspelled = self.spell.unknown(words)

        corrected_text = ""

        for word in words:
            corrected_word = self.spell.correction(word)

            if word not in misspelled or not corrected_word:
                corrected_word = word

            if corrected_word not in self.stopwords_set:
                p = self.morph.parse(corrected_word)[0]
                corrected_text += p.normal_form + " "

        return corrected_text.strip()

    def spell_ai(self, input_text):
        task_prefix = "Spell correct: "
        encoded = self.tokenizer(
            [task_prefix + input_text],
            padding="longest",
            max_length=self.MAX_INPUT,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        encoded = encoded.to(self.device)
        predicts = self.model.generate(**encoded)
        corrected_text = self.tokenizer.batch_decode(
            predicts, skip_special_tokens=True
        )[0]
        return corrected_text
