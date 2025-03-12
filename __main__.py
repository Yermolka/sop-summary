from summary import Summary, SummaryType
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-f", "--filename", type=str, default="data.csv")
    parser.add_argument("-o", "--output", type=str, default="summary.csv")
    parser.add_argument("-t", "--threads", type=int, default=4)
    parser.add_argument("--summary-type", type=SummaryType, default=SummaryType.BY_TEACHER)

    args = parser.parse_args()

    summary = Summary(args.filename, args.summary_type, args.threads, args.output)
    summary.summarize()
