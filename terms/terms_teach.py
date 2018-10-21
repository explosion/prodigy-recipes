# coding: utf8
from __future__ import unicode_literals

import prodigy
from prodigy.components.db import connect
from prodigy.components.sorters import Probability
from prodigy.util import split_string, set_hashes
import spacy
from spacy.tokens import Doc


# Recipe decorator with argument annotations: (description, argument type,
# shortcut, type / converter function called on value before it's passed to
# the function). Descriptions are also shown when typing --help.
@prodigy.recipe('terms.teach',
    dataset=("The dataset to use", "positional", None, str),
    vectors=("Loadable spaCy model with word vectors", "positional", None, str),
    seeds=("One or more comma-separated seed terms", "option", "o", split_string)
)
def terms_teach(dataset, vectors, seeds):
    """
    Bootstrap a terminology list with word vectors and seeds terms. Prodigy
    will suggest similar terms based on the word vectors, and update the
    target vector accordingly.
    """
    # Connect to the database using the settings from prodigy.json and add the
    # seed terms to the dataset
    DB = connect()
    if dataset and dataset in DB:
        seed_tasks = [set_hashes({'text': s, 'answer': 'accept'}) for s in seeds]
        DB.add_examples(seed_tasks, datasets=[dataset])

    # Load the spaCy model with vectors
    nlp = spacy.load(vectors)

    # Create two Doc objects for the accepted and rejected terms
    accept_doc = Doc(nlp.vocab, words=seeds)
    reject_doc = Doc(nlp.vocab, words=[])
    score = 0

    def predict(term):
        """Score a term given the current accept_doc and reject_doc."""
        if len(accept_doc) == 0 and len(reject_doc) == 0:
            return 0.5
        # Use spaCy's .similarity() method to compare the term to the
        # accepted and rejected Doc
        accept_score = max(term.similarity(accept_doc), 0.0)
        reject_score = max(term.similarity(reject_doc), 0.0)
        score = accept_score / (accept_score + reject_score + 0.2)
        return max(score, 0.0)

    def update(answers):
        # Called whenever Prodigy receives new annotations
        nonlocal accept_doc, reject_doc, score
        accept_words = [t.text for t in accept_doc]
        reject_words = [t.text for t in reject_doc]
        for answer in answers:
            # Increase or decrease score depending on answer and update
            # list of accepted and rejected terms
            if answer['answer'] == 'accept':
                score += 1
                accept_words.append(answer['text'])
            elif answer['answer'] == 'reject':
                score -= 1
                reject_words.append(answer['text'])
        # Update the target documents in place
        accept_doc = Doc(nlp.vocab, words=accept_words)
        reject_doc = Doc(nlp.vocab, words=reject_words)

    def score_stream(stream):
        # Get all lexemes in the vocab and score them
        lexemes = [lex for lex in stream if lex.is_alpha and lex.is_lower]
        while True:
            seen = set(w.orth for w in accept_doc)
            seen.update(set(w.orth for w in reject_doc))
            lexemes = [w for w in lexemes if w.orth not in seen]
            by_score = [(predict(lex), lex) for lex in lexemes]
            by_score.sort(reverse=True)
            for _, term in by_score:
                score = predict(term)
                # Return (score, example) tuples for the scored terms
                yield score, {'text': term.text, 'meta': {'score': score}}

    # Sort the scored vocab by probability and return examples
    stream = Probability(score_stream(nlp.vocab))

    return {
        'view_id': 'text',          # Annotation interface to use
        'dataset': dataset,         # Name of dataset to save annotations
        'stream': stream,           # Incoming stream of examples
        'update': update,           # Update callback, called with answers
    }
