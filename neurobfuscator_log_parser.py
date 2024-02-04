import argparse
import ast
import collections
import copy
import gc
import inspect
import itertools
import math
import pathlib
import re
import typing

import natsort
import numpy
import torch
import torchinfo


PathType = typing.Union[str, pathlib.Path]

# default device to run the summary on, GPU is fine but it might be using too much VRAM
# a good GPU option is cuda:1 or cuda:2 since 0/3 
DEFAULT_TORCH_DEVICE = torch.device('cpu')
# these two variables contain the imports required by the code as well
# as the ast blocks required, so that they are not filtered when parsing
NEUROBFUSCATOR_MODELS_REQUIRED_IMPORT_NAMES = ('torch', 'math', 'numpy')
NEUROBFUSCATOR_MODELS_REQUIRED_CODE_BLOCKS = (ast.ClassDef, ast.Import)
# default separator between models in the output fil
NEUROBFUSCATOR_MODEL_SEPARATOR = '\n'
# default separator between layers in the same model
NEUROBFUSCATOR_LAYER_SEPARATOR = ' , '
# default separator between layer parameters in the same model
NEUROBFUSCATOR_LAYER_PARAMETER_SEPARATOR = ' '
# default separator for arguments in the same 
# this variable contains the regex pattern to be used for matching
# the model configuration
NEUROBFUSCATOR_FINAL_ACTION_REGEX_PATTERN = 'DEBUG Best Action \(Full\):(?P<model_dict>.+?)\n'
# the list of blacklisted keys for the models, after they are parsed from
# the logs
NEUROBFUSCATOR_BLACKLISTED_MODEL_LOG_KEYS = ('prune_list', 'fuse_list')
# extra arguments to be passed to the model init, in this case the number
# of input features
NEUROBFUSCATOR_EXTRA_MODEL_ARGS = {'input_features': 3072}
# the arguments to be saved for each torch Module
NEUROBFUSCATOR_LAYER_ARGUMENTS_TO_BE_SAVED = {
    torch.nn.Conv2d: (
        'in_channels',
        'out_channels',
        'kernel_size',
        'stride',
        'padding',
    ),
    torch.nn.MaxPool2d: (
        'kernel_size',
        'stride',
        'padding',
    ),
    torch.nn.Linear: (
        'in_features',
        'out_features',
    ),
    torch.nn.BatchNorm2d: (
        'num_features',
        'eps',
        'momentum',
    ),
}
# the default shape used for NeurObfuscator models as input
# it is 3072 as it is 3 * 32 * 32 from CIFAR-10
NEUROBFUSCATOR_MODEL_INPUT_SHAPE = (3072, )


class TrainValTestSplitRatioAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values.split(':')) != 3:
            parser.error(f"We require 3 ratios for train:val:test")
        if sum(float(v) for v in values.split(':')) != 1:
            parser.error(f"Sum of the 3 ratios must be 1")
        ratios = values.split(':')
        final_value = {'train_ratio': float(ratios[0]), 'val_ratio': float(ratios[1]), 'test_ratio': float(ratios[2])}
        setattr(namespace, self.dest, copy.deepcopy(final_value))


# this function parses the log, returning the match from the regex
def parse_log_final_action(
    path: PathType,
    regex: str,
    group_index: int = -1,
) -> typing.Optional[typing.Dict[str, typing.Any]]:
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
    dict_str = matches.groups()[group_index]

    # we convert the string to dict using ast.literal_eval
    dict_ = ast.literal_eval(dict_str)

    return dict_


# this function converts a path, the required code blocks and imports
# together with the input dict to a final model instance
def create_model(
    # the path to the module
    path: PathType,
    # the input dict to the model class
    model_input_dict: typing.Dict[str, typing.Any],
    # the keys to be blacklisted from the model_input_dict
    model_input_dict_blacklist: typing.Optional[
            typing.Sequence[str]
    ] = None,
    # the index of the class we are interested in, default is the
    # last one
    class_index: int = -1,
    # the extra required code_blocks, apart from ClassDef
    code_blocks: typing.Optional[typing.Sequence[ast.AST]] = None,
    # if code_blocks contains also ast.Import, this
    # variable filter the **allowed** import names
    # not the final aliases, but the main name, e.g.
    # torch.nn.functional as F matches for torch
    imports: typing.Optional[typing.Sequence[str]] = None,
) -> "torch.nn.Module":
    path = pathlib.Path(path)
    code_blocks = (ast.ClassDef, ) + tuple(code_blocks if code_blocks is not None else ())

    # we get the code of the module by parsing it
    module_code = ast.parse(path.read_text())

    # we remove the unnecessary code, leaving only the required code blocks
    body = []
    for code_block in module_code.body:
        # we check the code_block to be one of the allowed instances
        if isinstance(code_block, code_blocks):
            # if additionally it is also an ast.Import, **and** the filter
            # is not None
            if isinstance(code_block, ast.Import) and imports is not None:
                # we check if the import name starts with any of the names
                # provided in the imports
                allowed_import = filter(
                    code_block.names[0].name.startswith,
                    imports
                )
                # if any of them is True then we append the code_block
                if any(allowed_import):
                    body.append(code_block)
            # if the code_block is not an import or the filter is None
            # we append it directly
            else:
                body.append(code_block)
    # we overwrite the original body
    module_code.body = body

    # we execute the code to generate the classes
    # we add the dicts to provide globals/locals dict, and avoid
    # errors with imports and defs
    # we need to save the dict to retrieve the model
    # we use only one dict as globals are used for __builtins__ mostly,
    # while all the imports and definitions are in locals
    # so if in the globals we have no locals definition, it will not work
    exec_dict = {}
    exec(compile(source=module_code, filename=str(path), mode='exec'), exec_dict, exec_dict)

    # we get the name of the class indexed by class_index
    class_name = [
        block
        for block in module_code.body
        if isinstance(block, ast.ClassDef)
    ][class_index]

    # now we remove the blacklisted keys from the input dict
    # we create a copy of the dict
    model_input_dict = copy.deepcopy(model_input_dict)
    # we iterate over all the blacklisted keys and we pop them from
    # the dict, only if it is not None
    if model_input_dict_blacklist is not None:
        for blacklist_key in model_input_dict_blacklist:
                # we use None as default value, so that there is no KeyError
                # if the key is not found
                model_input_dict.pop(blacklist_key, None)

    # we get the name from the ClassDef object
    # since we are using globals/locals namespaces for exec
    # we need to run the model name using those namespaces as well
    # so we generate the string for exec to be run
    # so we generate class_name(key=value, ...) for all the keys/values in the dict
    model_instance_exec_string = class_name.name + '_inst = ' + class_name.name + "(" + ", ".join(f"{k}={v}" for k, v in model_input_dict.items()) + ")"
    exec(model_instance_exec_string, exec_dict, exec_dict)
    model_instance = exec_dict[class_name.name + '_inst']

    return model_instance


# this function generates the layer sequence following the representation
# produced by calling repr on the model
# however, the actual representation is not the order in which the modules
# are defined, but due to the execution
# hence, we need torchinfo to generate the actual execution path
# this function will be renamed to generate_layer_definition_sequence
# to reflect the definition order of the layer sequence
def generate_layer_definition_sequence(
    model: "torch.nn.Module",
    layer_separator: str,
    include_layer_arguments: bool = False,
    arguments_to_include: typing.Optional[typing.Dict[
        "torch.nn.Module",
        typing.Sequence[str]
    ]] = None,
    argument_separator: str = NEUROBFUSCATOR_LAYER_PARAMETER_SEPARATOR,
) -> str:
    string = ''
    module_strings = []
    # we iterate over all the children modules
    for module in model.children():
        # we get the name of the class and we convert it to lowercase
        module_name = module.__class__.__qualname__.lower()

        module_string = module_name
        # if we have to include also the arguments
        # we need to inspect the init and get the arguments
        if include_layer_arguments:
            # we need an ordered dict not to lose the order of parameters
            args_dict = collections.OrderedDict()
            # we get the signature from the __init__ of the class
            signature = inspect.signature(module.__class__)
            # we iterate over the parameter names in the signature
            for argument_name in signature.parameters.keys():
                is_argument_to_be_included = (
                    # we check the dict of arguments to include
                    # is not None
                    arguments_to_include is not None and 
                    # and that the current argument is in the list
                    # corresponding to the class of the current module
                    # we put empty default if the class is not defined
                    # in the filtering dict
                    # in this case no argument will be saved
                    argument_name in arguments_to_include.get(
                        module.__class__,
                        []
                    )
                )
                # if the module has the corresponding parameter
                if (
                    hasattr(module,  argument_name) and 
                    is_argument_to_be_included
                ):
                    # then we get its value
                    argument_value = getattr(module,  argument_name)
                    # additionally, if the value is a tuple
                    # it means it is a kernel_size/padding_size, etc.
                    if isinstance(argument_value, tuple):
                        # we check that all the values in the tuple
                        # are the same, by filtering them with a set
                        # that eliminates duplicates and checking that there
                        # is only 1 element in the final set, so all
                        # the values must be the same
                        assert len(set(argument_value)) == 1
                        # we select the first one to be used in the
                        # representation
                        argument_value = argument_value[0]
                    # we set the value in the args dict, converting it to str
                    # for later joining them together
                    args_dict[argument_name] = str(argument_value)
            # we join the argument values using argument_separator as divider
            joint_args = argument_separator.join(args_dict.values())
            # we join the module name with the arguments, separating them
            # with an argument_separator, if the joint_args is not empty
            if joint_args:
                module_string += argument_separator + joint_args
        # we append the current module string to the list
        # of module strings
        module_strings.append(module_string)

    return layer_separator.join(module_strings)


# generates the layer sequence in the execution order
def generate_layer_execution_sequence(
    model: "torch.nn.Module",
    input_shape: torch.Size,
    layer_separator: str,
    include_layer_arguments: bool = False,
    arguments_to_include: typing.Optional[typing.Dict[
        "torch.nn.Module",
        typing.Sequence[str]
    ]] = None,
    argument_separator: str = NEUROBFUSCATOR_LAYER_PARAMETER_SEPARATOR,
    torch_device: torch.device = DEFAULT_TORCH_DEVICE,
) -> str:
    string = ''
    module_strings = []
    # we generate the fake input data and convert the model
    # to cpu
    model = model.cpu()
    input_data = torch.randn(input_shape, device=torch_device)
    # we generate the summary
    # verbose=0 to disable automatic output
    summary = torchinfo.summary(model, input_data=input_data, verbose=0, device=torch_device)
    # we generetate the list of children, by filtering the
    # modules based on the is_leaf_layer attribute
    # NOTE: is_leaf_layer is set as not any(module.children())
    # so if the module is a sub-model it would return false,
    # skipping it
    # we can check for it to be the first and also be a root
    # so that we skip the first model definition for sure
    modules = [
        layer_info.module
        for i, layer_info in enumerate(summary.summary_list)
        if layer_info.is_leaf_layer
    ]
    # we iterate over all the children modules
    for module in modules:
        # we get the name of the class and we convert it to lowercase
        module_name = module.__class__.__qualname__.lower()

        module_string = module_name
        # if we have to include also the arguments
        # we need to inspect the init and get the arguments
        if include_layer_arguments:
            # we need an ordered dict not to lose the order of parameters
            args_dict = collections.OrderedDict()
            # we get the signature from the __init__ of the class
            signature = inspect.signature(module.__class__)
            # we iterate over the parameter names in the signature
            for argument_name in signature.parameters.keys():
                is_argument_to_be_included = (
                        # we check the dict of arguments to include
                        # is not None
                        arguments_to_include is not None and 
                        # and that the current argument is in the list
                        # corresponding to the class of the current module
                        # we put empty default if the class is not defined
                        # in the filtering dict
                        # in this case no argument will be saved
                        argument_name in arguments_to_include.get(
                            module.__class__,
                            []
                        )
                )
                # if the module has the corresponding parameter
                if (
                    hasattr(module,  argument_name) and 
                    is_argument_to_be_included
                ):
                    # then we get its value
                    argument_value = getattr(module,  argument_name)
                    # additionally, if the value is a tuple
                    # it means it is a kernel_size/padding_size, etc.
                    if isinstance(argument_value, tuple):
                        # we check that all the values in the tuple
                        # are the same, by filtering them with a set
                        # that eliminates duplicates and checking that there
                        # is only 1 element in the final set, so all
                        # the values must be the same
                        assert len(set(argument_value)) == 1
                        # we select the first one to be used in the
                        # representation
                        argument_value = argument_value[0]
                    # we set the value in the args dict, converting it to str
                    # for later joining them together
                    args_dict[argument_name] = str(argument_value)
            # we join the argument values using argument_separator as divider
            joint_args = argument_separator.join(args_dict.values())
            # we join the module name with the arguments, separating them
            # with an argument_separator, if the joint_args is not empty
            if joint_args:
                module_string += argument_separator + joint_args
        # we append the current module string to the list
        # of module strings
        module_strings.append(module_string)

    del modules, summary
    gc.collect()

    return layer_separator.join(module_strings)


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
        '--model-dir',
        metavar='PATH',
        action='store',
        type=pathlib.Path,
        help='the path which contains all the model Python files',
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
        '--target-file',
        metavar='FILE',
        action='store',
        type=pathlib.Path,
        help='the file where to save the parsed vanilla models',
        required=True,
    )
    parser.add_argument(
        '--parse-layer-arguments',
        action='store_true',
        help='to enable parsing the arguments of each layer in each model',
    )
    parser.add_argument(
        '--parse-obfuscated',
        action='store_true',
        help='to enable parsing the models for the obfuscated versions',
    )
    parser.add_argument(
        '--obfuscated-target-file',
        metavar='FILE',
        action='store',
        type=pathlib.Path,
        help='the file where to save the parsed obfuscated models',
    )
    parser.add_argument(
        '--skip-if-errors',
        action='store_true',
        help='if passed skips the model either because of parsing errors in the log or because of missing obf file',
    )
    parser.add_argument(
        '--split-train-val-test',
        metavar='TRAIN_RATIO:VAL_RATIO:TEST_RATIO',
        type=str,
        action=TrainValTestSplitRatioAction,
        help='the ratio for splitting into train, validation and test target files. if not used, it will not be split. the ratios must sum to 1.',
    )
    parser.add_argument(
        '--torch-device',
        metavar='DEVICE STRING',
        type=torch.device,
        help='the string defining the device on which the summary runs. defaults to cpu',
        default=DEFAULT_TORCH_DEVICE,
    )
    parser.add_argument(
        '--filter-repetitions',
        action='store_true',
        help='if enabled, removes the duplicate entries in the model list. if also parsing obfuscated model, each entry is the tuple (model, obfuscated_model) that will be removed from both',
    )
    # others can be added if required

    return parser


# to filter the log filename and get the model name
def get_model_filename_from_log_filename(
    log_filename: PathType,
    log_separator: str,
    model_indices: typing.Sequence[int],
    model_id_separator: str,
) -> str:
    log_filename = pathlib.Path(log_filename).with_suffix('').name

    log_filename_splits = log_filename.split(log_separator)
    model_splits = []
    for index in model_indices:
        model_splits.append(log_filename_splits[index])

    model_string = ''.join(model_splits)
    model_string = model_string.replace('model', 'model' + model_id_separator)

    return model_string


def main(args=None):
    parser = setup_argparser()

    namespace = parser.parse_args(args=args)

    log_path = namespace.log_path
    model_dir = namespace.model_dir

    model_strings_list = []
    obf_model_strings_list = []
    log_list = []

    for log in natsort.natsorted(log_path.glob(namespace.log_glob_pattern)):
        print(f"Parsing {log}...")
        args_dict = parse_log_final_action(log, regex=NEUROBFUSCATOR_FINAL_ACTION_REGEX_PATTERN)
        # if args_dict is None:
        #     args_dict = parse_log_intermediate_action(log, regex=NEUROBFUSCATOR_INTERMEDIATE_ACTION_REGEX_PATTERN)
        if args_dict is None and namespace.skip_if_errors:
            print(f"Skipping log {str(log)}, as there was no result match in the log")
            continue
        args_dict.update(NEUROBFUSCATOR_EXTRA_MODEL_ARGS)

        model_name = get_model_filename_from_log_filename(log, log_separator='_', model_indices=[-2], model_id_separator='_')

        model_files = [(model_dir / model_name).with_suffix('.py')]
        model_strings_lists = [model_strings_list]

        if namespace.parse_obfuscated:
            model_obf_file = (model_dir / (model_name + '_obf')).with_suffix('.py')

            if not model_obf_file.exists() and namespace.skip_if_errors:
                print(f"Skipping {str(model_obf_file)}, file does not exist")
                continue

            model_files.append(model_obf_file)
            model_strings_lists.append(obf_model_strings_list)

        for m_file, m_strings_list in zip(model_files, model_strings_lists):
            model = create_model(
                m_file,
                model_input_dict=args_dict,
                model_input_dict_blacklist=NEUROBFUSCATOR_BLACKLISTED_MODEL_LOG_KEYS,
                code_blocks=NEUROBFUSCATOR_MODELS_REQUIRED_CODE_BLOCKS,
                imports=NEUROBFUSCATOR_MODELS_REQUIRED_IMPORT_NAMES
            )

            model_sequence_string = generate_layer_execution_sequence(
                model=model,
                input_shape=NEUROBFUSCATOR_MODEL_INPUT_SHAPE,
                layer_separator=NEUROBFUSCATOR_LAYER_SEPARATOR,
                include_layer_arguments=namespace.parse_layer_arguments,
                arguments_to_include=NEUROBFUSCATOR_LAYER_ARGUMENTS_TO_BE_SAVED,
                argument_separator=NEUROBFUSCATOR_LAYER_PARAMETER_SEPARATOR,
                torch_device=namespace.torch_device
            )

            m_strings_list.append(model_sequence_string)
        
        log_list.append(log)

    target_files = []
    strings = []

    if namespace.filter_repetitions:
        if not namespace.parse_obfuscated:
            model_strings_list = list(dict.fromkeys(model_strings_list).keys())
        else:
            temp_list = list(dict.fromkeys(itertools.zip_longest(
                model_strings_list,
                obf_model_strings_list,
                fillvalue='',  
            )).keys())
            if not temp_list:
                model_strings_list, obf_model_strings_list = [], []
            else:
                model_strings_list, obf_model_strings_list = zip(*temp_list)

    if namespace.split_train_val_test:
        train_ratio = namespace.split_train_val_test['train_ratio']
        val_ratio = namespace.split_train_val_test['val_ratio']
        test_ratio = namespace.split_train_val_test['test_ratio']

        train_index = math.floor(train_ratio * len(model_strings_list))
        val_index = math.floor(val_ratio * len(model_strings_list)) + train_index

        train_model_strings = NEUROBFUSCATOR_MODEL_SEPARATOR.join(model_strings_list[:train_index])
        val_model_strings = NEUROBFUSCATOR_MODEL_SEPARATOR.join(model_strings_list[train_index:val_index])
        test_model_strings = NEUROBFUSCATOR_MODEL_SEPARATOR.join(model_strings_list[val_index:])

        train_target_file = namespace.target_file.with_suffix('.train' + ''.join(namespace.target_file.suffixes))
        val_target_file = namespace.target_file.with_suffix('.val' + ''.join(namespace.target_file.suffixes))
        test_target_file = namespace.target_file.with_suffix('.test' + ''.join(namespace.target_file.suffixes))

        target_files.extend([train_target_file, val_target_file, test_target_file])
        strings.extend([train_model_strings, val_model_strings, test_model_strings])

        if namespace.parse_obfuscated:
            train_index = math.floor(train_ratio * len(obf_model_strings_list))
            val_index = math.floor(val_ratio * len(obf_model_strings_list)) + train_index
            
            train_obf_model_strings = NEUROBFUSCATOR_MODEL_SEPARATOR.join(obf_model_strings_list[:train_index])
            val_obf_model_strings = NEUROBFUSCATOR_MODEL_SEPARATOR.join(obf_model_strings_list[train_index:val_index])
            test_obf_model_strings = NEUROBFUSCATOR_MODEL_SEPARATOR.join(obf_model_strings_list[val_index:])

            train_obf_target_file = namespace.obfuscated_target_file.with_suffix('.train' + ''.join(namespace.obfuscated_target_file.suffixes))
            val_obf_target_file = namespace.obfuscated_target_file.with_suffix('.val' + ''.join(namespace.obfuscated_target_file.suffixes))
            test_obf_target_file = namespace.obfuscated_target_file.with_suffix('.test' + ''.join(namespace.obfuscated_target_file.suffixes))

            target_files.extend([train_obf_target_file, val_obf_target_file, test_obf_target_file])
            strings.extend([train_obf_model_strings, val_obf_model_strings, test_obf_model_strings])
    else:
        model_strings = NEUROBFUSCATOR_MODEL_SEPARATOR.join(model_strings_list)
        target_file = namespace.target_file
        target_files.append(target_file)
        strings.append(model_strings)

        if namespace.parse_obfuscated:
            obf_model_strings = NEUROBFUSCATOR_MODEL_SEPARATOR.join(obf_model_strings_list)
            target_files.append(namespace.obfuscated_target_file)
            strings.append(obf_model_strings)

    for target, string in zip(target_files, strings):
        target.parent.mkdir(parents=True, exist_ok=True)

        target.write_text(string)

    
if __name__ == '__main__':
    main()        




