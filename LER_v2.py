import itertools
import pathlib
import random
import sys
import typing

LevenshteinInput = typing.Union[typing.Sequence[str], str]


def levenshtein_distance_recursive(str1: LevenshteinInput, str2: LevenshteinInput) -> int:
    if not len(str1):
        return len(str2)
    elif not len(str2):
        return len(str1)
    elif str1[0] == str2[0]:
        return levenshtein_distance(str1=str1[1:], str2=str2[1:])
    else:
        return 1 + min(
            levenshtein_distance(str1=str1[1:], str2=str2),
            levenshtein_distance(str1=str1, str2=str2[1:]),
            levenshtein_distance(str1=str1[1:], str2=str2[1:])
        )


def levenshteinDistance(s1: LevenshteinInput, s2: LevenshteinInput) -> int:
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def normalized_levenshtein_distance(ground_truth: LevenshteinInput, predictions: LevenshteinInput) -> float:
    editing_distance = levenshteinDistance(ground_truth, predictions)
    return editing_distance / len(ground_truth)


def sequence_levenshtein_distance(ground_truth: typing.Sequence[LevenshteinInput], predictions: typing.Sequence[LevenshteinInput]) -> float:
    return sum(normalized_levenshtein_distance(gt, p) for gt, p in itertools.zip_longest(ground_truth, predictions, fillvalue=[]))


def der(ground_truth: typing.Sequence[LevenshteinInput], predictions: typing.Sequence[LevenshteinInput]) -> float:
    s = 0
    for gt_seq, p_seq in itertools.zip_longest(ground_truth, predictions, fillvalue=[]):
        for gt, p in itertools.zip_longest(gt_seq, p_seq, fillvalue=0):
            s += abs(p - gt) / (gt if gt else 1)

    return s


def ler_metric_on_generic_sequence(ground_truth: typing.Sequence[LevenshteinInput], predictions: typing.Sequence[LevenshteinInput]) -> float:
    # ground_truth_layer_sequence = [hash(l[0] if len(l) > 0 else random.random()) for l in ground_truth]
    # predictions_layer_sequence = [hash(l[0] if len(l) > 0 else random.random()) for l in predictions]
    ground_truth_layer_sequence = [l[0] if len(l) > 0 else "" for l in ground_truth]
    predictions_layer_sequence = [l[0] if len(l) > 0 else "" for l in predictions]

    # print(ground_truth_layer_sequence)

    ler = normalized_levenshtein_distance(ground_truth=ground_truth_layer_sequence, predictions=predictions_layer_sequence)

    return ler

# in neurobfuscator, they use the LER for computing the editing distance
# between a sequence of layers which is encoded in numbers, e.g. 0001, 0100,
# with 1 meaning conv2d and 0 meaning linear
# then they repeat the process for the corresponding dimensions, so they
# have ((1, 2), (2, )) where the first is a conv2d and the second is a linear
# against ((3, 2), (4, )), and they compute the normalized distance wrt to the ground
# truth as sum all of them together for each element and for each sequence
def ler_der_metric(ground_truth: typing.Sequence[LevenshteinInput], predictions: typing.Sequence[LevenshteinInput]) -> float:
    ground_truth_layer_parameter_sequence = [l[1:] for l in ground_truth]
    predictions_layer_parameter_sequence = [l[1:] for l in predictions]

    der_value = der(ground_truth=ground_truth_layer_parameter_sequence, predictions=predictions_layer_parameter_sequence)




if __name__ == "__main__":
    file1 = pathlib.Path(sys.argv[1])
    file2 = pathlib.Path(sys.argv[2])
    split = sys.argv[3]

    for l1, l2 in zip(file1.read_text().splitlines(), file2.read_text().splitlines()):
        print(ler_metric_on_generic_sequence(ground_truth=[[y.strip() for y in x.strip().split()] for x in l1.split(split)], predictions=[[y.strip() for y in x.strip().split()] for x in l2.split(split)]))