import argparse
import pathlib
import typing

import numpy

import neurobfuscator_log_parser

LAYER_NAME_TO_INT_MAP: typing.Dict[str, int] = {
    'conv2d': 0,
    'conv': 0,
    'linear': 1,
    'fc': 1,
    'adaptiveavgpool2d': 2,
    'maxpool2d': 2,
    'avgpool2d': 2,
    'batchnorm1d': 3,
    'batchnorm2d': 3,
    'depthconv': 4,
    'relu': 5,
    'pointconv': 6,
    'add': 7,
    'logsoftmax': 8,
    'softmax': 8
}
LAYER_NAMES_TO_BE_SKIPPED = ['relu']

def make_label_from_model(
    model: "torch.nn.Module",
    input_shape: "torch.Size",
    conversion_dict: typing.Dict[str, int],
    keys_to_be_skipped: typing.Optional[typing.Sequence[str]] = None,
) -> numpy.array:
    # we get the layer sequence as a string with the names of the layers
    # separated by a space
    layer_seq = neurobfuscator_log_parser.generate_layer_execution_sequence(
        model=model,
        input_shape=input_shape,
        layer_separator=' ',
        include_layer_arguments=False
    )

    # we go through each name splitting the string by space
    array = []
    for layer_name in layer_seq.split(' '):
        # if the layer name is not to be skipped then we append the corresponding number
        # in the array
        if keys_to_be_skipped is None or layer_name not in keys_to_be_skipped:
            array.append(conversion_dict[layer_name])

    # we return the numpy array conversion of the integer list
    return numpy.array(array)


def setup_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--target-npy-dir',
            metavar='PATH',
            action='store',
            type=pathlib.Path,
            help='the path where to save the .npy files',
            required=True,
    )
    parser.add_argument(
            '--model-dir',
            metavar='PATH',
            action='store',
            type=pathlib.Path,
            help='the path which contains all the model Python files',
            required=True,
    )
    parser.add_argument(
            '--skip-errors',
            action='store_true',
            help='if passed it skips the erroring models by printing the error message, otherwise it propagates the error',
    )
    parser.add_argument(
            '--delete-on-error',
            action='store_true',
            help='if passed deletes the models which leads to exceptions, default is no deletion',
    )
    parser.add_argument(
            '--skip-if-npy-exists',
            action='store_true',
            help='if passed skips the npy generation for a certain model if the file already exists',
    )
    # others can be added if required

    return parser


def main(args=None):
    parser = setup_argparser()

    namespace = parser.parse_args(args=args)

    model_dir = namespace.model_dir
    for model_path in model_dir.glob('*.py'):
        npy_file = namespace.target_npy_dir / model_path.with_suffix('.npy').name

        if namespace.skip_if_npy_exists and npy_file.exists():
            print(f"Skipping model {str(model_path)}, corresponding .npy file {str(npy_file)} already exists!")
            continue

        model = neurobfuscator_log_parser.create_model(
            model_path,
            model_input_dict=neurobfuscator_log_parser.NEUROBFUSCATOR_EXTRA_MODEL_ARGS,
            code_blocks=neurobfuscator_log_parser.NEUROBFUSCATOR_MODELS_REQUIRED_CODE_BLOCKS,
            imports=neurobfuscator_log_parser.NEUROBFUSCATOR_MODELS_REQUIRED_IMPORT_NAMES,
        )

        try:
            npy_label = make_label_from_model(
                model=model,
                input_shape=neurobfuscator_log_parser.NEUROBFUSCATOR_MODEL_INPUT_SHAPE,
                conversion_dict=LAYER_NAME_TO_INT_MAP,
                keys_to_be_skipped=LAYER_NAMES_TO_BE_SKIPPED,
            )
        except RuntimeError as e:
            if namespace.skip_errors:
                print(f"Skipping model {str(model_path)}, caught error: {repr(e)}")
            # if the option is not enabled we raise the error as this is not
            # supposed to be caught
            else:
                raise e
            
            if namespace.delete_on_error:
                print(f"Deleting model {str(model_path)}")
                model_path.unlink()

        print(f"Creating .npy file {str(npy_file)}")

        npy_file.parent.mkdir(parents=True, exist_ok=True)

        numpy.save(str(npy_file), npy_label)


if __name__ == '__main__':
    main()