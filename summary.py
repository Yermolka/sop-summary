import pandas as pd
from enum import IntFlag
from asyncio import run, gather


# Флаги, работают вместе через побитовое ИЛИ
# Например, по преподавателям и по годам - 3 (BY_TEACHER | BY_YEAR)
class SummaryType(IntFlag):
    BY_TEACHER = 1
    BY_YEAR = 2
    BY_COURSE = 4

class Summary:
    # Ключ - значения группировки, например "Седашов 2024-2025 НИС"
    # Значение - список строк из СОПа, например ["лучший", "Z", "V"]
    data: dict[str, list[str]] = {}
    
    # для поддержания порядка ключей
    data_keys: list[str] = []

    results: dict[str, str] = {}

    def __init__(self, filename: str, summary_type: SummaryType, workers: int, output: str):
        self.df = pd.read_csv(filename)
        self.df["summary"] = None

        self.summary_type = summary_type
        self.workers = workers
        self.output = output

    def summarize(self) -> str:
        print(f"\n{'='*80}")
        print("Running summarize...")
        print(f"{'='*80}\n")

        self._prepare_data()
        run(self._summarize())

    def _prepare_data(self) -> None:
        for idx, row in self.df.iterrows():
            if idx >= 10_000:
                break

            program = str(row["ОП"]) or ""
            teacher = str(row["ФИО преподавателя"]) or ""
            year = str(row["Семестр"].split()[0]) or ""
            course = str(row["Дисциплина"]) or ""

            key = program + " "
            if SummaryType.BY_TEACHER in self.summary_type:
                key += teacher + " "
            if SummaryType.BY_YEAR in self.summary_type:
                key += year + " "
            if SummaryType.BY_COURSE in self.summary_type:
                key += course + " "

            text_key = "text_without_stopwords"
            self.data.setdefault(key, []).append(str(row[text_key]))

        self.data_keys = list(self.data.keys())

    async def _summarize(self) -> str:
        results = await gather(*[self._worker_thread(i) for i in range(self.workers)])
        for result in results:
            for key, summary in result.items():
                self.results[key] = summary

        self._save_results()
    
    async def _worker_thread(self, idx: int) -> dict[str, str]:
        keys_per_worker = len(self.data) // self.workers
        keys_per_worker += 1 if len(self.data) % self.workers != 0 else 0

        start = idx * keys_per_worker
        end = start + keys_per_worker
        keys = self.data_keys[start:end]

        results = {}

        for key in keys:
            results[key] = self._ai_magic(self.data[key])

        return results

    def _ai_magic(self, texts: list[str]) -> str:
        return "".join(texts)

    def _save_results(self) -> None:
        ordered_result_keys = list(sorted(self.results.keys()))

        results = pd.DataFrame([[key, self.results[key]] for key in ordered_result_keys], columns=["key", "summary"])
        results.to_csv(self.output, index=False)
