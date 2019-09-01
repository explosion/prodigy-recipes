# coding: utf8
from __future__ import unicode_literals
import sys

import prodigy
from prodigy.core import recipe_args
from prodigy.components.db import connect
from prodigy.components.sorters import Probability
from prodigy.util import log, prints, split_string, set_hashes
import requests
import spacy
import srsly


@prodigy.recipe('phrases.teach',
    dataset=recipe_args["dataset"],
    seeds=("One or more comma-separated seed terms", "option", "se", split_string),
    threshold=("Similarity threshold for sense2vec", "option", "t", float),
    batch_size=("Similarity threshold for sense2vec", "option", "bs", int),
    resume=("Resume from existing phrases dataset", "flag", "R", bool)
)
def phrases_teach(dataset, seeds, threshold=0.85, batch_size=5, resume=False):
    """
    Bootstrap a terminology list with word vectors and seeds terms. Prodigy
    will suggest similar terms based on the word vectors, and update the
    target vector accordingly.
    """

    DB = connect()
    seed_tasks = [set_hashes({"text": s, "answer": "accept"}) for s in seeds]
    DB.add_examples(seed_tasks, datasets=[dataset])

    accept_phrases = seeds
    reject_phrases = []

    seen = set(accept_phrases)
    sensed = set()

    if resume:
        prev = DB.get_dataset(dataset)
        prev_accept = [eg["text"] for eg in prev if eg["answer"] == "accept"]
        prev_reject = [eg["text"] for eg in prev if eg["answer"] == "reject"]
        accept_phrases += prev_accept
        reject_phrases += prev_reject

        seen.update(set(accept_phrases))
        seen.update(set(reject_phrases))

    def sense2vec(phrase, threshold):
        """Call sense2vec API to get similar "senses" (phrases)"""
        res = requests.post('https://api.explosion.ai/sense2vec/find', {
            "sense": "auto",
            "word": phrase
        })
        results = res.json()["results"]
        output = []
        for r in results:
            if r["score"] > threshold or len(output) <= 10:
                output.append((r["score"], r["text"]))

        return output

    def update(answers):
        """Updates accept_phrases so that the stream can find new phrases"""
        for answer in answers:
            if answer['answer'] == 'accept':
                accept_phrases.append(answer['text'])
            elif answer['answer'] == 'reject':
                reject_phrases.append(answer['text'])
    
    def get_stream():
        """Continue querying sense2vec whenever we get a new phrase and presenting
        examples to the user with a similarity above the threshold parameter"""
        while True:
            seen.update(set([rp.lower() for rp in reject_phrases]))
            for p in accept_phrases:
                if p.lower() not in sensed:
                    sensed.add(p.lower())
                    for score, phrase in sense2vec(p, threshold):
                        if phrase.lower() not in seen:
                            seen.add(phrase.lower())
                            yield score, {"text": phrase, 'meta': {'score': score}}

    stream = Probability(get_stream())

    return {
        'view_id': 'text',
        'dataset': dataset,
        'stream': stream,
        'update': update,
        'config': {
            "batch_size": batch_size
        }
    }


@prodigy.recipe(
    "phrases.to-patterns",
    dataset=recipe_args["dataset"],
    label=recipe_args["label"],
    output_file=recipe_args["output_file"],
)
def to_patterns(dataset=None, label=None, output_file=None):
    """
    Convert a list of seed phrases to a list of match patterns that can be used
    with ner.match. If no output file is specified, each pattern is printed
    so the recipe's output can be piped forward to ner.match.

    This is pretty much an exact copy of terms.to-patterns.
    The pattern for each example is just split on whitespace so instead of:

        {"label": "SHOE_BRAND", "pattern": [{"LOWER": "new balance"}]}


    which won't match anything you'll get:

        {"label": "SHOE_BRAND", "pattern": [{"LOWER": "new"}, {"LOWER": "balance"}]}
    """
    if label is None:
        prints(
            "--label is a required argument",
            "This is the label that will be assigned to all patterns "
            "created from terms collected in this dataset. ",
            exits=1,
            error=True,
        )

    DB = connect()

    def get_pattern(term, label):
        return {"label": label, "pattern": [{"lower": t.lower()} for t in term["text"].split()]}

    log("RECIPE: Starting recipe terms.to-patterns", locals())
    if dataset is None:
        log("RECIPE: Reading input terms from sys.stdin")
        terms = (srsly.json_loads(line) for line in sys.stdin)
    else:
        if dataset not in DB:
            prints("Can't find dataset '{}'".format(dataset), exits=1, error=True)
        terms = DB.get_dataset(dataset)
        log(
            "RECIPE: Reading {} input terms from dataset {}".format(len(terms), dataset)
        )
    if output_file:
        patterns = [
            get_pattern(term, label) for term in terms if term["answer"] == "accept"
        ]
        log("RECIPE: Generated {} patterns".format(len(patterns)))
        srsly.write_jsonl(output_file, patterns)
        prints("Exported {} patterns".format(len(patterns)), output_file)
    else:
        log("RECIPE: Outputting patterns")
        for term in terms:
            if term["answer"] == "accept":
                print(srsly.json_dumps(get_pattern(term, label)))
