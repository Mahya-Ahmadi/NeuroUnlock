import argparse
import json
import pathlib

DEFAULT_BASE_LOG_NAME = "obf_model_{:0>10d}.log"
DEFAULT_LINE_BEGINNING = "[{'"
DEFAULT_LINE_TEMPLATE = "DEBUG Best Action (Full):{}\n"


def setup_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--logs',
        metavar='LOG_PATH',
        nargs='+',
        type=pathlib.Path,
        help='the logs to parse',
    )

    parser.add_argument(
        '--dest-dir',
        metavar='DESTINATION_DIR',
        action='store',
        type=pathlib.Path,
        help='the destination folder for the parsed logs',
        required=True,
    )

    parser.add_argument(
        "--base-log-name",
        metavar='BASE_LOG_NAME',
        action='store',
        help='the base name to be used for each log file, must contain "{}"',
        default=DEFAULT_BASE_LOG_NAME,
    )

    parser.add_argument(
        "--line-template",
        metavar='BASE_LINE_TEMPLATE',
        action='store',
        help='the template for printing out each line, must contain "{}"',
        default=DEFAULT_LINE_TEMPLATE,
    )
    parser.add_argument(
        "--line-beginning",
        metavar='LINE_BEGINNING',
        action='store',
        help='the beginning of the lines that need to be selected',
        default=DEFAULT_LINE_BEGINNING,
    )
    return parser


def main(args=None):
    parser = setup_argparser()

    namespace = parser.parse_args(args=args)

    namespace.dest_dir.mkdir(parents=True, exist_ok=True)

    for log in namespace.logs:
        log_content = log.read_text()
        counter = 0
        for line in log_content.splitlines():
            if line.startswith(namespace.line_beginning):
                line = line.replace("'", '"')
                parsed_line = json.loads(line)
                for model_obf_list in parsed_line:
                    (namespace.dest_dir / namespace.base_log_name.format(counter)).write_text(namespace.line_template.format(model_obf_list))
                    counter += 1


if __name__ == "__main__":
    main()