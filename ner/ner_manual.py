# coding: utf8
from __future__ import unicode_literals

import prodigy
from prodigy.components.loaders import JSONL
from prodigy.components.preprocess import add_tokens
from prodigy.util import split_string
import spacy


# Recipe decorator with argument annotations: (description, argument type,
# shortcut, type / converter function called on value before it's passed to
# the function). Descriptions are also shown when typing --help.
@prodigy.recipe('ner.manual',
    dataset=("The dataset to use", "positional", None, str),
    spacy_model=("The base model", "positional", None, str),
    source=("The source data as a JSONL file", "positional", None, str),
    label=("One or more comma-separated labels", "option", "l", split_string),
    exclude=("Names of datasets to exclude", "option", "e", split_string)
)
def ner_manual(dataset, spacy_model, source, label=None, exclude=None):
    """
    Mark spans manually by token. Requires only a tokenizer and no entity
    recognizer, and doesn't do any active learning.
    """
    # Load the spaCy model for tokenization
    nlp = spacy.load(spacy_model)

    # Load the stream from a JSONL file and return a generator that yields a
    # dictionary for each example in the data.
    stream = JSONL(source)

    # Tokenize the incoming examples and add a "tokens" property to each
    # example. Also handles pre-defined selected spans. Tokenization allows
    # faster highlighting, because the selection can "snap" to token boundaries.
    stream = add_tokens(nlp, stream)

    return {
        'view_id': 'ner_manual', # Annotation interface to use
        'dataset': dataset,      # Name of dataset to save annotations
        'stream': stream,        # Incoming stream of examples
        'exclude': exclude,      # List of dataset names to exclude
        'config': {              # Additional config settings, mostly for app UI
            'lang': nlp.lang,
            'label': ', '.join(label) if label is not None else 'all',
            'labels': label      # Selectable label options
        }
    }
