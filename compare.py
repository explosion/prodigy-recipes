# coding: utf8
from __future__ import unicode_literals

from pathlib import Path
import random

from prodigy.core import recipe, recipe_args
from prodigy.components.printers import get_compare_printer
from prodigy.util import read_jsonl, log


@recipe('compare',
        dataset=recipe_args['dataset'],
        a_file=("JSONL file for system responses", "positional", None, Path),
        b_file=("JSONL file for baseline responses", "positional", None, Path),
        no_random=("Don't randomise which annotation is shown as correct",
                   "flag", "NR", bool),
        diff=recipe_args['diff'])
def compare(dataset, a_file, b_file, no_random=False, diff=False):
    """
    Compare output of two models and randomly assign A/B categories.
    """
    log("RECIPE: Starting recipe compare", locals())
    a_questions = read_jsonl(a_file)
    b_questions = read_jsonl(b_file)

    return {
        'dataset': dataset,
        'view_id': 'diff' if diff is True else 'compare',
        'stream': get_questions(a_questions, b_questions, not no_random),
        'update': None,
        'progress': None,
        'on_exit': get_compare_printer(Path(a_file).name, Path(b_file).name)
    }


def get_questions(a_questions, b_questions, randomize):
    a_questions = {a['id']: a for a in a_questions}
    b_questions = {b['id']: b for b in b_questions}
    for id_, a in a_questions.items():
        if id_ not in b_questions:
            continue
        question = {'id': id_, 'input': a['input']}
        a = a['output']
        b = b_questions[id_]['output']
        if a == b:
            continue
        if randomize and random.random() >= 0.5:
            question['accept'], question['reject'] = b, a
            question['mapping'] = {'B': 'accept', 'A': 'reject'}
        else:
            question['accept'], question['reject'] = a, b
            question['mapping'] = {'A': 'accept', 'B': 'reject'}
        yield question
