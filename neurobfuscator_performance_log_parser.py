import argparse
import ast
import collections
import copy
import gc
import inspect
import itertools
import json
import math
import os
import pathlib
import re
import typing

import numpy
import torch
import torchinfo


PathType = os.PathLike

# this regex matches the clean latency at the beginning of the log
NEUROBFUSCATOR_UNOBFUSCATED_LATENCY_REGEX_PATTERN = 'DEBUG Clean Cycle is (?P<unobfuscated_latency>.+?)\n'
# this regex matches the best latency at each generation
NEUROBFUSCATOR_GENERATION_OBFUSCATED_LATENCY_REGEX_PATTERN = ", its Latency is: (?P<generation_obfuscated_latency>.+?),"
# this regex matches the final/best latency if available
NEUROBFUSCATOR_BEST_OBFUSCATED_LATENCY_REGEX_PATTERN = "DEBUG Best Latency: (?P<final_obfuscated_latency>.+?)\n"


# this function parses the log, returning the match from the regex
def parse_log_performance(
    path: PathType,
    regex: str,
    group_index: int = -1,
) -> int:
    path = pathlib.Path(path)

    # we create the pattern matcher
    pattern = re.compile(regex)

    # we generate all the matches, by reading into the 
    # search of the pattern the whole log file
    matches = pattern.search(path.read_text())
    # we return None if there is no match
    if matches is None:
        return None

    # we return the get the string representing the dict
    # to be returned, using group_index
    groups = matches.groups()
    try:
        element_str = groups[group_index]
    except IndexError:
        return None

    # we convert the string to dict using ast.literal_eval
    element = int(ast.literal_eval(element_str))

    return element


def setup_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--log-path',
        metavar='PATH',
        action='store',
        type=pathlib.Path,
        help='the path which contains the logs',
        required=True,
    )
    parser.add_argument(
        '--log-glob-pattern',
        metavar='PATTERN',
        action='store',
        type=str,
        default='*.log',
        help='the glob pattern to use for the logs',
    )
    parser.add_argument(
        '--target-json-file',
        metavar='FILE',
        action='store',
        type=pathlib.Path,
        help='the file where to save the JSON representation of the parsed logs will be stored',
        required=True,
    )
    # others can be added if required

    return parser


def main(args=None):
    parser = setup_argparser()

    namespace = parser.parse_args(args=args)

    log_path = namespace.log_path
    
    log_dict = dict()
    for log in log_path.glob(namespace.log_glob_pattern):
        current_log_dict = dict()
        # for each log we parse unobfuscated latency, generation and best obfuscated latency
        unobfuscated_latency = parse_log_performance(log, NEUROBFUSCATOR_UNOBFUSCATED_LATENCY_REGEX_PATTERN)
        current_log_dict["unobfuscated_latency"] = unobfuscated_latency

        generation_obfuscated_latency = 0
        counter = 0
        current_log_dict["generation_obfuscated_latency"] = []
        while generation_obfuscated_latency is not None:
            generation_obfuscated_latency = parse_log_performance(log, NEUROBFUSCATOR_GENERATION_OBFUSCATED_LATENCY_REGEX_PATTERN, group_index=counter)
            current_log_dict["generation_obfuscated_latency"].append(generation_obfuscated_latency)
            counter += 1

        best_obfuscated_latency = parse_log_performance(log, NEUROBFUSCATOR_BEST_OBFUSCATED_LATENCY_REGEX_PATTERN)
        current_log_dict["best_obfuscated_latency"] = best_obfuscated_latency

        log_dict[str(pathlib.Path(log).absolute())] = current_log_dict

    json_dump = json.dumps(log_dict, indent=4, sort_keys=True)

    namespace.target_json_file.parent.mkdir(exist_ok=True, parents=True)
    namespace.target_json_file.write_text(json_dump)

    
if __name__ == '__main__':
    main()        




