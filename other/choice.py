# coding: utf8
from __future__ import unicode_literals

import prodigy
from prodigy.components.loaders import JSONL
from prodigy.util import split_string


def add_options(stream, options):
    """Helper function to add options to every task in a stream."""
    options = [{'id': option, 'text': option} for option in options]
    for task in stream:
        task['options'] = options
        yield task


# Recipe decorator with argument annotations: (description, argument type,
# shortcut, type / converter function called on value before it's passed to
# the function). Descriptions are also shown when typing --help.
@prodigy.recipe('choice',
    dataset=("The dataset to use", "positional", None, str),
    source=("The source data as a JSONL file", "positional", None, str),
    options=("One or more comma-separated options", "option", "o", split_string),
    multiple=("Allow multiple choice", "flag", "M", bool)
)
def choice(dataset, source=None, options=None, multiple=False):
    """
    Annotate data with multiple-choice options. The annotated examples will
    have an additional property `"accept": []` mapping to the ID(s) of the
    selected option(s).
    """
    # Load the stream from a JSONL file and return a generator that yields a
    # dictionary for each example in the data.
    stream = JSONL(source)

    # Add the options to all examples in the stream
    stream = add_options(stream, options)

    return {
        'view_id': 'choice',    # Annotation interface to use
        'dataset': dataset,     # Name of dataset to save annotations
        'stream': stream,       # Incoming stream of examples
        'config': {             # Additional config settings
            # Allow multiple choice if flag is set
            'choice_style': 'multiple' if multiple else 'single',
            # Automatically accept and "lock in" selected answers if only
            # single choice is allowed
            'choice_auto_accept': False if multiple else True
        }
    }
