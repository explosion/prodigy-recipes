# coding: utf8
from __future__ import unicode_literals

import prodigy
from prodigy.models.ner import EntityRecognizer
from prodigy.components.preprocess import add_tokens
from prodigy.components.db import connect
from prodigy.util import split_string
import spacy


# Recipe decorator with argument annotations: (description, argument type,
# shortcut, type / converter function called on value before it's passed to
# the function). Descriptions are also shown when typing --help.
@prodigy.recipe('ner.silver-to-gold',
    silver_dataset=("Existing dataset with binary annotations", "positional", None, str),
    gold_dataset=("Name of dataset to save new annotations", "positional", None, str),
    spacy_model=("The base model", "positional", None, str),
    label=("One or more comma-separated labels", "option", "l", split_string)
)
def ner_silver_to_gold(silver_dataset, gold_dataset, spacy_model, label=[]):
    """
    Take an existing "silver" dataset with binary accept/reject annotations,
    merge the annotations to find the best possible analysis given the
    constraints defined in the annotations, and manually edit it to create
    a perfect and complete "gold" dataset.
    """
    # Connect to the database using the settings from prodigy.json, check
    # that the silver dataset exists and load it
    DB = connect()
    if silver_dataset not in DB:
        raise ValueError("Can't find dataset '{}'.".format(silver_dataset))
    silver_data = DB.get_dataset(silver_dataset)

    # Load the spaCy model
    nlp = spacy.load(spacy_model)
    if not label:
        # Get the labels from the model by looking at the available moves, e.g.
        # B-PERSON, I-PERSON, L-PERSON, U-PERSON
        ner = nlp.get_pipe('ner')
        moves = ner.move_names
        label = [move.split('-')[1] for move in moves if move[0] in ('B', 'I', 'L', 'U')]
        label = sorted(set(label))

    # Initialize Prodigy's entity recognizer model, which uses beam search to
    # find all possible analyses and outputs (score, example) tuples
    model = EntityRecognizer(nlp, label=label)

    # Merge all annotations and find the best possible analyses
    stream = model.make_best(silver_data)

    # Tokenize the incoming examples and add a "tokens" property to each
    # example. Also handles pre-defined selected spans. Tokenization allows
    # faster highlighting, because the selection can "snap" to token boundaries.
    stream = add_tokens(nlp, stream)

    return {
        'view_id': 'ner_manual', # Annotation interface to use
        'dataset': gold_dataset, # Name of dataset to save annotations
        'stream': stream,        # Incoming stream of examples
        'config': {              # Additional config settings, mostly for app UI
            'lang': nlp.lang,
            'labels': label     # Selectable label options
        }
    }
