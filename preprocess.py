import pandas as pd
import pymorphy3
import re
import nltk
from nltk.corpus import stopwords
from spellchecker import SpellChecker
from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast
import torch

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
        self.data["text"] = self.data["Комментарий студента"].apply(self.text_preprocess)
        self.data["lemmatized_text"] = self.data["text"].apply(self.lemmatize)
        self.data["lemmatized_text_without_stopwords"] = self.data["lemmatized_text"].apply(self.remove_stopwords)
        self.data["text_without_stopwords"] = self.data["text"].apply(self.remove_stopwords)

        self.data.to_excel("preprocessed.xlsx", index=False)
        self.data.to_csv("preprocessed.csv", index=False)

        # Применяем модель на небольшой выборке
        data1 = self.data.sample(n=100).copy()
        data1["Исправленный комментарий"] = data1["Комментарий студента"].apply(self.spell_ai)

        data1.to_excel("preprocessed_part_corrected.xlsx", index=False)
        data1.to_csv("preprocessed_part_corrected.csv", index=False)

    def text_to_lower(self, text: str) -> str:
        """
        Принимает на вход текст, возвращает текст в нижнем регистре.
        """
        return text.lower()

    def remove_punctuation(self, text: str) -> str:
        """
        Принимает на вход текст, вовзращает текст, в котором символы перехода на
        новую строку, символы табуляции и знаки препинания заменены на пробелы.
        """
        cleaned_text = text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
        return re.sub(r"[^\w\s]", " ", cleaned_text)

    def remove_extra_spaces(self, text: str) -> str:
        """
        Принимает на вход текст, возвращает его без лишних пробелов.
        """
        return re.sub(r"\s+", " ", text).strip()

    def text_preprocess(self, text: str) -> str:
        """
        Принимает на вход текст, возвращает текст в нижнем регистре без знаков
        препинания и без лишних пробелов.
        """
        return self.remove_extra_spaces(
            self.remove_punctuation(self.text_to_lower(text))
        )

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

    def remove_stopwords(self, text: str) -> str:
        """
        Принимает на вход текст в нижнем регистре без знаков препинания, возвращает
        текст с удаленными стоп-словами.
        """
        # Написать перед вызовом функции строчки:
        # nltk.download('stopwords')
        # stopwords_set = set(stopwords.words('russian'))

        words = text.split(" ")
        return " ".join([word for word in words if word not in self.stopwords_set])

    def lemmatize(self, text: str) -> str:
        """
        Принимает на вход текст после препроцессинга, приводит все слова в нем к начальным формам.
        """
        # Перед вызовом функции нужно объявить morph, написать строчку:
        # morph = pymorphy3.MorphAnalyzer()

        text = text.split(" ")
        res = ""
        for word in text:
            p = self.morph.parse(word)[0]
            res += p.normal_form + " "
        return res.strip()

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
