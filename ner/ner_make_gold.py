# coding: utf8
from __future__ import unicode_literals

import prodigy
from prodigy.components.loaders import JSONL
from prodigy.components.preprocess import add_tokens
from prodigy.util import split_string, set_hashes
import spacy
import copy


def make_tasks(nlp, stream, labels):
    """Add a 'spans' key to each example, with predicted entities."""
    # Process the stream using spaCy's nlp.pipe, which yields doc objects.
    # If as_tuples=True is set, you can pass in (text, context) tuples.
    texts = ((eg['text'], eg) for eg in stream)
    for doc, eg in nlp.pipe(texts, as_tuples=True):
        task = copy.deepcopy(eg)
        spans = []
        for ent in doc.ents:
            # Continue if predicted entity is not selected in labels
            if labels and ent.label_ not in labels:
                continue
            # Create span dict for the predicted entitiy
            spans.append({
                'token_start': ent.start,
                'token_end': ent.end - 1,
                'start': ent.start_char,
                'end': ent.end_char,
                'text': ent.text,
                'label': ent.label_
            })
        task['spans'] = spans
        # Rehash the newly created task so that hashes reflect added data
        task = set_hashes(task)
        yield task


# Recipe decorator with argument annotations: (description, argument type,
# shortcut, type / converter function called on value before it's passed to
# the function). Descriptions are also shown when typing --help.
@prodigy.recipe('ner.make-gold',
    dataset=("The dataset to use", "positional", None, str),
    spacy_model=("The base model", "positional", None, str),
    source=("The source data as a JSONL file", "positional", None, str),
    label=("One or more comma-separated labels", "options", "l", split_string),
    exclude=("Names of datasets to exclude", "option", "e", split_string)
)
def ner_make_gold(dataset, spacy_model, source, label=None, exclude=None):
    """
    Create gold-standard data by correcting a model's predictions manually.
    """
    # Load the spaCy model
    nlp = spacy.load(spacy_model)

    # Load the stream from a JSONL file and return a generator that yields a
    # dictionary for each example in the data.
    stream = JSONL(source)

    # Tokenize the incoming examples and add a "tokens" property to each
    # example. Also handles pre-defined selected spans. Tokenization allows
    # faster highlighting, because the selection can "snap" to token boundaries.
    stream = add_tokens(nlp, stream)

    # Add the entities predicted by the model to the tasks in the stream
    stream = make_tasks(nlp, stream, label)

    return {
        'view_id': 'ner_manual', # Annotation interface to use
        'dataset': dataset,      # Name of dataset to save annotations
        'stream': stream,        # Incoming stream of examples
        'exclude': exclude,      # List of dataset names to exclude
        'config': {              # Additional config settings, mostly for app UI
            'lang': nlp.lang,
            'label': ', '.join(label) if label is not None else 'all',
            'labels': label     # Selectable label options
        }
    }
