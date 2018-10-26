# coding: utf8
from __future__ import unicode_literals

import prodigy
from prodigy.components.loaders import JSONL
from prodigy.models.matcher import PatternMatcher
from prodigy.components.db import connect
from prodigy.util import split_string
import spacy


# Recipe decorator with argument annotations: (description, argument type,
# shortcut, type / converter function called on value before it's passed to
# the function). Descriptions are also shown when typing --help.
@prodigy.recipe('ner.match',
    dataset=("The dataset to use", "positional", None, str),
    spacy_model=("The base model", "positional", None, str),
    source=("The source data as a JSONL file", "positional", None, str),
    patterns=("Optional match patterns", "option", "p", str),
    exclude=("Names of datasets to exclude", "option", "e", split_string),
    resume=("Resume from existing dataset and update matcher accordingly", "flag", "R", bool)
)
def ner_match(dataset, spacy_model, source, patterns=None, exclude=None,
              resume=False):
    """
    Suggest phrases that match a given patterns file, and mark whether they
    are examples of the entity you're interested in. The patterns file can
    include exact strings or token patterns for use with spaCy's `Matcher`.
    """
    # Load the spaCy model
    nlp = spacy.load(spacy_model)

    # Initialize the pattern matcher and load in the JSONL patterns
    matcher = PatternMatcher(nlp).from_disk(patterns)

    if resume:
        # Connect to the database using the settings from prodigy.json
        DB = connect()
        if dataset and dataset in DB:
            # Get the existing annotations and update the matcher
            existing = DB.get_dataset(dataset)
            matcher.update(existing)

    # Load the stream from a JSONL file and return a generator that yields a
    # dictionary for each example in the data.
    stream = JSONL(source)

    # Apply the matcher to the stream, which returns (score, example) tuples.
    # Filter out the scores to only yield the examples for annotations.
    stream = (eg for score, eg in matcher(stream))

    return {
        'view_id': 'ner',       # Annotation interface to use
        'dataset': dataset,     # Name of dataset to save annotations
        'stream': stream,       # Incoming stream of examples
        'exclude': exclude,     # List of dataset names to exclude
        'config': {             # Additional config settings, mostly for app UI
            'lang': nlp.lang
        }
    }
