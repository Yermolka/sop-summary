from summary import Summary, SummaryType
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(
        usage="python3 __main__.py -f <filename> -o <output> -t -y -c",
        description="Generate a summary of the data in the file"
    )

    parser.add_argument("-f", "--filename", type=str, default="data.csv", help="Path to the CSV file with the data")
    parser.add_argument("-o", "--output", type=str, default="summary.csv", help="Path to the output CSV file")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads to use")

    parser.add_argument("-t", "--teacher", action="store_true", help="Group by teacher")
    parser.add_argument("-y", "--year", action="store_true", help="Group by year")
    parser.add_argument("-c", "--course", action="store_true", help="Group by course")

    args = parser.parse_args()

    if not any([args.teacher, args.year, args.course]):
        parser.error("At least one of the following arguments is required: -t, -y, -c")

    summary_type = 0
    if args.teacher:
        summary_type |= SummaryType.BY_TEACHER
    if args.year:
        summary_type |= SummaryType.BY_YEAR
    if args.course:
        summary_type |= SummaryType.BY_COURSE

    summary = Summary(args.filename, summary_type, args.threads, args.output)
    summary.summarize()
