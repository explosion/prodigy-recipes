# coding: utf8
from __future__ import unicode_literals

import prodigy
from prodigy.components.loaders import JSONL
from prodigy.util import split_string
from collections import Counter


# Recipe decorator with argument annotations: (description, argument type,
# shortcut, type / converter function called on value before it's passed to
# the function). Descriptions are also shown when typing --help.
@prodigy.recipe('mark',
    dataset=("The dataset to use", "positional", None, str),
    source=("The source data as a JSONL file", "positional", None, str),
    view_id=("ID of annotation interface", "option", "o", str),
    exclude=("Names of datasets to exclude", "option", "e", split_string)
)
def mark(dataset, source, view_id, exclude=None):
    """
    Click through pre-prepared examples, with no model in the loop.
    """
    counts = Counter()

    # Load the stream from a JSONL file and return a generator that yields a
    # dictionary for each example in the data.
    stream = JSONL(source)

    def on_load(controller):
        # Check if current dataset is available in database. The on_load
        # callback receives the controller as an argument, which exposes the
        # database via controller.db
        if dataset in controller.db:
            examples = controller.db.get_dataset(dataset)
            for eg in examples:
                # Update counts with existing answers
                counts[eg['answer']] += 1

    def receive_answers(answers):
        for eg in answers:
            # Update counts with new answers
            counts[eg['answer']] += 1

    def on_exit(controller):
        # Output the total annotation counts
        print('Accept:', counts['accept'])
        print('Reject:', counts['reject'])
        print('Ignore:', counts['ignore'])
        print('Total: ', sum(counts.values()))

    return {
        'view_id': view_id,         # Annotation interface to use
        'dataset': dataset,         # Name of dataset to save annotations
        'stream': stream,           # Incoming stream of examples
        'update': receive_answers,  # Update callback, called with answers
        'on_load': on_load,         # Called on first load
        'on_exit': on_exit          # Called when Prodigy server is stopped
    }
