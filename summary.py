from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM, T5Tokenizer, EncoderDecoderModel
import torch
import pandas as pd
from enum import IntFlag
from tqdm import tqdm
from time import perf_counter
from typing import Generator
import logging
import json
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

    model_name = "IlyaGusev/rubert_telegram_headlines"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = EncoderDecoderModel.from_pretrained(model_name, device_map="cuda:0")
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="cuda:0")
    LIMIT_LINES_TEST = 1_000
    CHUNK_SIZE = 512

    def __init__(self, filename: str, summary_type: SummaryType, workers: int, output: str):
        self.df = pd.read_csv(filename)
        self.df["summary"] = None

        self.summary_type = summary_type
        self.workers = workers
        self.output = output

        logging.basicConfig(
            filename=f"summary_{self.model_name.replace('/', '_')}.log",
            filemode="a",
            format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)

        self.process_times: list[tuple[float, int]] = [] # time, text length    

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
        self.logger.info(f"Prepared {len(self.data_keys)} keys")

    def _summarize(self) -> str:
        for key in tqdm(self.data_keys):
            start_time = perf_counter()
            self.results[key] = self._ai_magic(self.data[key])
            end_time = perf_counter()
            self.process_times.append((end_time - start_time, len("".join(self.data[key]))))

            self.logger.info(f"Processed {key} in {self.process_times[-1][0]} seconds, {self.process_times[-1][1]} symbols")

        total_process_time = (sum(time for time, _ in self.process_times), sum(length for _, length in self.process_times))
        self.logger.info(f"Total process time: {total_process_time[0]} seconds, {total_process_time[1]} symbols")

        self.process_times.append(total_process_time)

        with open(f"process_times_{self.model_name.replace('/', '_')}.json", "w") as f:
            json.dump(self.process_times, f)

    def _chunked_text(self, text: str) -> Generator[str, None, None]:
        # text = "summary: " + text #для utrobinmv/t5_summary_en_ru_zh_base_2048
        # text = '[{0:.1g}] '.format(0.2) + text

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        self.logger.info(f"Tokens: {len(tokens)}")
        return (tokens[i:i+self.CHUNK_SIZE] for i in range(0, len(tokens), self.CHUNK_SIZE))

    def _ai_magic(self, texts: list[str]) -> str:
        article_text = "".join(texts)
        summaries = []
        for chunk in self._chunked_text(article_text):
            summaries.append(self._ai_magic_inner(chunk))

        if len(summaries) > 1:
            return self._ai_magic(summaries)
        
        return summaries[0]
    
    def _ai_magic_inner(self, input_tokens: list[int]) -> str:
        input_ids = torch.tensor(input_tokens).unsqueeze(0)
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda:0")

        output_ids = self.model.generate(
            input_ids=input_ids,
            no_repeat_ngram_size=4
        )[0]

        return self.tokenizer.decode(output_ids, skip_special_tokens=True)

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
