# coding: utf8
from __future__ import unicode_literals

import prodigy
from prodigy.components.loaders import JSONL
from prodigy.models.textcat import TextClassifier
from prodigy.models.matcher import PatternMatcher
from prodigy.components.sorters import prefer_uncertain
from prodigy.util import combine_models, split_string
import spacy


# Recipe decorator with argument annotations: (description, argument type,
# shortcut, type / converter function called on value before it's passed to
# the function). Descriptions are also shown when typing --help.
@prodigy.recipe('textcat.teach',
    dataset=("The dataset to use", "positional", None, str),
    spacy_model=("The base model", "positional", None, str),
    source=("The source data as a JSONL file", "positional", None, str),
    label=("One or more comma-separated labels", "option", "l", split_string),
    patterns=("Optional match patterns", "option", "p", str),
    exclude=("Names of datasets to exclude", "option", "e", split_string),
    long_text=("Enable long-text classification mode", "flag", "L", bool)
)
def textcat_teach(dataset, spacy_model, source, label=None, patterns=None,
                  exclude=None, long_text=False):
    """
    Collect the best possible training data for a text classification model
    with the model in the loop. Based on your annotations, Prodigy will decide
    which questions to ask next.
    """
    # Load the stream from a JSONL file and return a generator that yields a
    # dictionary for each example in the data.
    stream = JSONL(source)

    # Load the spaCy model
    nlp = spacy.load(spacy_model)

    # Initialize Prodigy's text classifier model, which outputs
    # (score, example) tuples
    model = TextClassifier(nlp, label, long_text=long_text)

    if patterns is None:
        # No patterns are used, so just use the model to suggest examples
        # and only use the model's update method as the update callback
        predict = model
        update = model.update
    else:
        # Initialize the pattern matcher and load in the JSONL patterns.
        # Set the matcher to not label the highlighted spans, only the text.
        matcher = PatternMatcher(nlp, prior_correct=5., prior_incorrect=5.,
                                 label_span=False, label_task=True)
        matcher = matcher.from_disk(patterns)
        # Combine the NER model and the matcher and interleave their
        # suggestions and update both at the same time
        predict, update = combine_models(model, matcher)

    # Use the prefer_uncertain sorter to focus on suggestions that the model
    # is most uncertain about (i.e. with a score closest to 0.5). The model
    # yields (score, example) tuples and the sorter yields just the example
    stream = prefer_uncertain(predict(stream))

    return {
        'view_id': 'classification', # Annotation interface to use
        'dataset': dataset,          # Name of dataset to save annotations
        'stream': stream,            # Incoming stream of examples
        'update': update,            # Update callback, called with batch of answers
        'exclude': exclude,          # List of dataset names to exclude
        'config': {                  # Additional config settings, mostly for app UI
            'lang': nlp.lang,
            'label': ', '.join(label) if label is not None else 'n/a'
        }
    }
