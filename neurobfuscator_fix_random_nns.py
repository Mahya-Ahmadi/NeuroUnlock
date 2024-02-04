import argparse
import pathlib
import typing


MODEL_FILENAME_TEMPLATE = "model_{model_id}.py"
MODEL_NAME_TEMPLATE = "custom_cnn_{model_id}"

MODEL_START_TOKEN = "# Model starts here"
MODEL_STOP_TOKEN = "# Model ends here"


# this custom Action is used in the parser to check that the starting model id
# is at least equal to the minimum
# Answer: https://stackoverflow.com/a/18700817
class ModelIDAction(argparse.Action):
    MINIMUM_ALLOWABLE_ID = 1
    def __call__(self, parser, namespace, values, option_string=None):
        if values < self.MINIMUM_ALLOWABLE_ID:
            parser.error(f"Minimum value for {option_string} is {self.MINIMUM_ALLOWABLE_ID}")

        setattr(namespace, self.dest, values)


class ModelTemplateFileAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # in this case we check the path to be readable
        try:
            with values.open('r') as f:
                pass
        except FileNotFoundError:
            parser.error(f"The {option_string} file must exist and be readable")
        else:
            setattr(namespace, self.dest, values)


MODEL_CLASS_DEF_LINE_TEMPLATE = "class {model_name}(torch.nn.Module):"
MODEL_INIT_DEF_LINE_TEMPLATE = "def __init__(self, input_features, reshape = True, widen_list = None, decompo_list = None, dummy_list = None, deepen_list = None, skipcon_list = None, kerneladd_list = None):"
MODEL_SUPER_LINE_TEMPLATE = "super().__init__();self.reshape = reshape;self.widen_list = widen_list;self.decompo_list = decompo_list;self.dummy_list = dummy_list;self.deepen_list = deepen_list;self.skipcon_list = skipcon_list;self.kerneladd_list = kerneladd_list;"


# the following functions are used to check the presence of a certain line
# if the presence is found the line is returned, otherwise None is returned

# to check for a class definition, we check for the line to start with class and end with :
# this covers only single-line class definitions, but it should be enough for our use-case
def class_def_checker(line: str) -> typing.Union[str, None]:
    # we strip the line of starting/trailing space characters
    line = line.strip()
    if line.startswith('class') and line.endswith(':'):
        return line
    return None


# for checking for a super line, we check whether the line starts with super and contains __init__
# this because we are only interested in the super call made for the init of torch.nn.Module
def super_checker(line: str) -> typing.Union[str, None]:
    # we strip the line of starting/trailing space characters
    line = line.strip()
    if line.startswith('super') and '__init__' in line:
        return line
    return None


# for checking for a init definition line, we check whether the line starts with def and contains __init__
# as well as ending with :, forcing the __init__ to be a single line, but it should be enough in our use-case
def init_def_checker(line: str) -> typing.Union[str, None]:
    # we strip the line of starting/trailing space characters
    line = line.strip()
    if line.startswith('def') and '__init__' in line and line.endswith(':'):
        return line
    return None


MODEL_SUBSTITUION_DICT = {
    class_def_checker: MODEL_CLASS_DEF_LINE_TEMPLATE,
    init_def_checker: MODEL_INIT_DEF_LINE_TEMPLATE,
    super_checker: MODEL_SUPER_LINE_TEMPLATE,
}


# we get the source of a model definition from the whole source code, by cutting the source between two
# well-known identifiers
def get_source_model_class_definition(
    source_file: pathlib.Path,
    start_token: str,
    stop_token: str,
) -> str:
    # we read the soruce code from the source file
    source_code = source_file.read_text()

    # here we split the whole source code with the start token
    # since we suppose there is only one start token available in the
    # source code, the model definition will follow the start token
    model_source_start_split = source_code.split(start_token)[-1]
    # then we repeat a similar operation for the stop token, but in this case
    # the stop token will be before the stop token
    model_source = model_source_start_split.split(stop_token)[0]

    # we return the source code with start and stop token being re-added
    return start_token + model_source + stop_token


# we fix the model source, by changing model name, 
# as well as updating the super init call and the 
# init arguments
def fix_model_source(
    model_source: str,
    substitution_dict: typing.Dict[typing.Callable[[str], typing.Union[str, None]], str],
    template_format_kwargs: typing.Dict[str, str],
) -> str:
    # we split the source code in lines
    model_source_lines = model_source.split('\n')

    matching_dict = {key: None for key in substitution_dict.keys()}
    # we get the lines corresponding to the different definitions
    for line in model_source_lines:
        # we iterate over each function and value
        # if each is still None we try to update it
        for fn, value in matching_dict.items():
            if value is None:
                matching_dict[fn] = fn(line)
        # if all of them are not None we can stop
        if all(value is not None for value in matching_dict.values()):
            break
    
    # we check all the values have been found
    for fn, value in matching_dict.items():
        if value is None:
            raise RuntimeError(f"The following function did not find a match: {fn.__name__}")

    # we match the keys in the matched lines with the substitution templates
    for key in matching_dict.keys():
        # in each template we substitute the formatting dict
        # e.g. a dict like {'model_name': 'custom_cnn_4', ...}
        formatted_template = substitution_dict[key].format(**template_format_kwargs)
        # we replace the matching line with the formatted template in the source
        model_source = model_source.replace(matching_dict[key], formatted_template)

    return model_source


# we add the fixed model source to the template
# in this case we need a model_source for model_source_placeholder
# as well as model_name
# we create the target directory and file if required and we write the
# final source
def add_model_source_to_template(
    template: pathlib.Path,
    target_path: pathlib.Path,
    model_source: str,
    model_name: str,
) -> pathlib.Path:
    template_source = template.read_text()

    template_source = template_source.format(model_source_placeholder=model_source, model_name=model_name)

    target_path.parent.mkdir(parents=True, exist_ok=True)

    target_path.write_text(template_source)

    return target_path


def setup_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--target-dir',
            metavar='PATH',
            action='store',
            type=pathlib.Path,
            help='the path where to save the fixed model files',
            required=True,
    )
    parser.add_argument(
            '--model-dir',
            metavar='PATH',
            action='store',
            type=pathlib.Path,
            help='the path which contains all the model files to be fixed',
            required=True,
    )
    parser.add_argument(
            '--model-template-file',
            metavar='PATH',
            action=ModelTemplateFileAction,
            type=pathlib.Path,
            help='the path to the model template',
            required=True,
    )
    parser.add_argument(
            '--model-starting-index',
            metavar='ID',
            action=ModelIDAction,
            type=int,
            default=1,
            help='the starting index for the model values, it is the index of the model_{id}.py file, which is also the index of the model custom_cnn_{id - 1}. default: 1',
            required=True,
    )
    parser.add_argument(
            '--model-name-id-difference',
            metavar="INT",
            action='store',
            type=int,
            required=True,
            default=1,
            help='required difference between the model filename id and the model name id, as in {model_name_id}={model_filename_id}-{difference}, default is 1',
    )

    # others can be added if required

    return parser


def main(args=None):
    parser = setup_argparser()

    namespace = parser.parse_args(args=args)

    model_dir = namespace.model_dir
    model_template = namespace.model_template_file

    for model_id, model_path in enumerate(model_dir.glob('*.py'), start=namespace.model_starting_index):
        # the model filename requires the model id
        model_filename = MODEL_FILENAME_TEMPLATE.format(model_id=model_id)
        # for the model name we need the model id - 1
        # seq_obfuscator uses custom_cnn_{id-1}
        # dim_obfuscator uses custom_cnn_{id}
        model_name = MODEL_NAME_TEMPLATE.format(model_id=model_id - namespace.model_name_id_difference)

        # this is the final target path for the current model
        model_target = namespace.target_dir / model_filename

        model_source = get_source_model_class_definition(source_file=model_path, start_token=MODEL_START_TOKEN, stop_token=MODEL_STOP_TOKEN)

        model_source = fix_model_source(model_source=model_source, substitution_dict=MODEL_SUBSTITUION_DICT, template_format_kwargs={'model_name': model_name})

        model_target = add_model_source_to_template(template=model_template, target_path=model_target, model_source=model_source, model_name=model_name)

        print(f"{str(model_path)} converted to the fixed model in {str(model_target)}")
        


if __name__ == '__main__':
    main()
    