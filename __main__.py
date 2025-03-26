from typing import Literal
from summary import Summary, SummaryType
from tone_analysis import ToneAnalysis
from preprocess import Preprocess
from time import sleep

if __name__ == "__main__":
    # Перед запуском:
    #    pip install -r requirements.txt
    #    Поместить файл SOP_report_fio_id.xlsx в папку со скриптом
    #    Выставить нужные настройки ниже
    #    python __main__.py ИЛИ python .


    # Общие настройки
    task: Literal["preprocess", "summary", "tone_analysis"] = "preprocess"
    print("Task: ", task)
    sleep(2)

    # Настройки для summary
    # Модель 0 - cointegrated/rut5-base-absum, быстрая
    # Модель 1 - IlyaGusev/rut5-base-sum-gazeta, медленная, но более полная и похожая на человеческий язык
    model_idx: int = 0

    # Группировка по
    group_by_teacher: bool = True
    group_by_year: bool = False
    group_by_course: bool = False

    match task:
        case "preprocess":
            preprocess = Preprocess("SOP_report_fio_id.xlsx")
            preprocess.preprocess()
        case "summary":
            if (
                not any([group_by_teacher, group_by_year, group_by_course])
                and task == "summary"
            ):
                print("Group by set to default: by teacher")
                sleep(2)
                group_by_teacher = True

            summary_type = 0
            if group_by_teacher:
                summary_type |= SummaryType.BY_TEACHER
            if group_by_year:
                summary_type |= SummaryType.BY_YEAR
            if group_by_course:
                summary_type |= SummaryType.BY_COURSE
            summary = Summary("preprocessed.csv", summary_type, model_idx)
            summary.summarize()
        case "tone_analysis":
            tone_analysis = ToneAnalysis("preprocessed.csv")
            tone_analysis.tone_analysis()
