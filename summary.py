from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import pandas as pd
from enum import IntFlag
from tqdm import tqdm

torch.cuda.is_available()

# Флаги, работают вместе через побитовое ИЛИ
# Например, по преподавателям и по годам - 3 (BY_TEACHER | BY_YEAR)
class SummaryType(IntFlag):
    BY_TEACHER = 1
    BY_YEAR = 2
    BY_COURSE = 4

class Summary:
    # Ключ - значения группировки, например "Седашов 2024-2025 НИС"
    # Значение - список строк из СОПа, например ["лучший", "Z", "V"]
    data: dict[tuple, list[str]] = {}
    
    # для поддержания порядка ключей
    data_keys: list[tuple] = []

    results: dict[tuple, str] = {}

    model_name = "IlyaGusev/rut5_base_sum_gazeta"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="cuda:0")

    LIMIT_LINES_TEST = 200

    def __init__(self, filename: str, summary_type: SummaryType, workers: int, output: str):
        self.df = pd.read_csv(filename)
        self.df["summary"] = None

        self.summary_type = summary_type
        self.workers = workers
        self.output = output

        if torch.cuda.is_available():
            print(torch.cuda.get_device_name(0))
            self.model = self.model.to("cuda:0")
        else:
            print("Using CPU")

    def summarize(self) -> str:
        print(f"\n{'='*80}")
        print("Running summarize...")
        print(f"{'='*80}\n")

        self._prepare_data()
        self._summarize()
        self._save_results()

    def _prepare_data(self) -> None:
        for idx, row in self.df.iterrows():
            if idx >= self.LIMIT_LINES_TEST:
                break

            program = str(row["ОП"]) or ""
            teacher = str(row["ФИО преподавателя"] or "")
            year = str(row["Семестр"].split()[0] or "")
            course = str(row["Дисциплина"] or "")

            keys = [program]
            if SummaryType.BY_TEACHER in self.summary_type:
                keys.append(teacher)
            if SummaryType.BY_YEAR in self.summary_type:
                keys.append(year)
            if SummaryType.BY_COURSE in self.summary_type:
                keys.append(course)

            text_key = "text"
            self.data.setdefault(tuple(keys), []).append(str(row[text_key]))

        self.data_keys = list(self.data.keys())

    def _summarize(self) -> str:
        for key in tqdm(self.data_keys):
            self.results[key] = self._ai_magic(self.data[key])

    def _ai_magic(self, texts: list[str]) -> str:
        article_text = "".join(texts)
        input_ids = self.tokenizer(
            [article_text],
            max_length=600,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )["input_ids"]

        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda:0")

        output_ids = self.model.generate(
            input_ids=input_ids,
            no_repeat_ngram_size=4
        )[0]

        summary = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return summary


    def _save_results(self) -> None:
        ordered_result_keys = list(sorted(self.results.keys()))

        keys = ["program"]
        if SummaryType.BY_TEACHER in self.summary_type:
            keys.append("teacher")
        if SummaryType.BY_YEAR in self.summary_type:
            keys.append("year")
        if SummaryType.BY_COURSE in self.summary_type:
            keys.append("course")

        results = pd.DataFrame([[*result_key, self.results[result_key], len(self.data[result_key])] for result_key in ordered_result_keys], columns=[*keys, "summary", "based_on_len"])
        results.to_csv(self.output, index=False)
