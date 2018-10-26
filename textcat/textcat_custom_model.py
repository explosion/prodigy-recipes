# coding: utf8
from __future__ import unicode_literals

import prodigy
from prodigy.components.loaders import JSONL
from prodigy.components.sorters import prefer_uncertain
from prodigy.util import split_string
import random


class DummyModel(object):
    # This is a dummy model to help illustrate how to use Prodigy with a model
    # in the loop. It currently "predicts" random numbers – but you can swap
    # it out for any model of your choice, for example a text classification
    # model implementation using PyTorch, TensorFlow or scikit-learn.

    def __init__(self, labels=None):
        # The model can keep arbitrary state – let's use a simple random float
        # to represent the current weights
        self.weights = random.random()
        self.labels = labels

    def __call__(self, stream):
        for eg in stream:
            # Score the example with respect to the current weights and
            # assign a label
            eg['label'] = random.choice(self.labels)
            score = (random.random() + self.weights) / 2
            yield (score, eg)

    def update(self, answers):
        # Update the model weights with the new answers. This method receives
        # the examples with an added "answer" key that either maps to "accept",
        # "reject" or "ignore".
        self.weights = random.random()


# Recipe decorator with argument annotations: (description, argument type,
# shortcut, type / converter function called on value before it's passed to
# the function). Descriptions are also shown when typing --help.
@prodigy.recipe('textcat.custom-model',
    dataset=("The dataset to use", "positional", None, str),
    source=("The source data as a JSONL file", "positional", None, str),
    label=("One or more comma-separated labels", "option", "l", split_string)
)
def textcat_custom_model(dataset, source, label=[]):
    """
    Use active learning-powered text classification with a custom model. To
    demonstrate how it works, this demo recipe uses a simple dummy model that
    "precits" random scores. But you can swap it out for any model of your
    choice, for example a text classification model implementation using
    PyTorch, TensorFlow or scikit-learn.
    """
    # Load the stream from a JSONL file and return a generator that yields a
    # dictionary for each example in the data.
    stream = JSONL(source)

    # Load the dummy model
    model = DummyModel(labels=label)

    # Use the prefer_uncertain sorter to focus on suggestions that the model
    # is most uncertain about (i.e. with a score closest to 0.5). The model
    # yields (score, example) tuples and the sorter yields just the example
    stream = prefer_uncertain(model(stream))

    # The update method is called every time Prodigy receives new answers from
    # the web app. It can be used to update the model in the loop.
    update = model.update

    return {
        'view_id': 'classification', # Annotation interface to use
        'dataset': dataset,          # Name of dataset to save annotations
        'stream': stream,            # Incoming stream of examples
        'update': update,            # Update callback, called with batch of answers
        'config': {                  # Additional config settings, mostly for app UI
            'label': ', '.join(label)
        }
    }
