# coding: utf8
from __future__ import unicode_literals

import random
import mmh3
import json
import spacy
import copy

from .compare import get_questions as get_compare_questions
from ..models.ner import EntityRecognizer, merge_spans
from ..models.matcher import PatternMatcher
from ..components import printers
from ..components.db import connect
from ..components.preprocess import split_sentences, split_spans, add_tokens
from ..components.sorters import prefer_uncertain
from ..components.loaders import get_stream
from ..components.filters import filter_tasks
from ..core import recipe, recipe_args
from ..util import split_evals, get_labels, get_print, combine_models
from ..util import export_model_data, set_hashes, log, prints
from ..util import INPUT_HASH_ATTR, TASK_HASH_ATTR


DB = connect()


@recipe('ner.match',
        dataset=recipe_args['dataset'],
        spacy_model=recipe_args['spacy_model'],
        source=recipe_args['source'],
        api=recipe_args['api'],
        loader=recipe_args['loader'],
        patterns=recipe_args['patterns'],
        exclude=recipe_args['exclude'])
def match(dataset, spacy_model, patterns, source=None, api=None, loader=None,
          exclude=None):
    """
    Suggest phrases that match a given patterns file, and mark whether they
    are examples of the entity you're interested in. The patterns file can
    include exact strings, regular expressions, or token patterns for use with
    spaCy's `Matcher` class.
    """
    log("RECIPE: Starting recipe ner.match", locals())
    # Create the model, using a pre-trained spaCy model.
    model = PatternMatcher(spacy.load(spacy_model)).from_disk(patterns)
    log("RECIPE: Created PatternMatcher using model {}".format(spacy_model))
    if dataset is not None and dataset in DB:
        existing = DB.get_dataset(dataset)
        log("RECIPE: Updating PatternMatcher with {} examples from dataset {}"
            .format(len(existing), dataset))
        model.update(existing)
    stream = get_stream(source, api=api, loader=loader, rehash=True,
                        dedup=True, input_key='text')
    return {
        'view_id': 'ner',
        'dataset': dataset,
        'stream': (eg for _, eg in model(stream)),
        'exclude': exclude
    }


@recipe('ner.teach',
        dataset=recipe_args['dataset'],
        spacy_model=recipe_args['spacy_model'],
        source=recipe_args['source'],
        api=recipe_args['api'],
        loader=recipe_args['loader'],
        label=recipe_args['entity_label'],
        patterns=recipe_args['patterns'],
        exclude=recipe_args['exclude'])
def teach(dataset, spacy_model, source=None, api=None, loader=None,
          label=None, patterns=None, exclude=None):
    """
    Collect the best possible training data for a named entity recognition
    model with the model in the loop. Based on your annotations, Prodigy will
    decide which questions to ask next.
    """
    log("RECIPE: Starting recipe ner.teach", locals())
    # Initialize the stream, and ensure that hashes are correct, and examples
    # are deduplicated.
    stream = get_stream(source, api=api, loader=loader, rehash=True,
                        dedup=True, input_key='text')
    # Create the model, using a pre-trained spaCy model.
    nlp = spacy.load(spacy_model, disable=['parser', 'tagger'])
    log("RECIPE: Creating EntityRecognizer using model {}".format(spacy_model))
    model = EntityRecognizer(nlp, label=label)
    if label is not None and patterns is None:
        log("RECIPE: Making sure all labels are in the model", label)
        for l in label:
            if not model.has_label(l):
                prints("Can't find label '{}' in model {}"
                       .format(l, spacy_model),
                       "ner.teach will only show entities with one of the "
                       "specified labels. If a label is not available in the "
                       "model, Prodigy won't be able to propose entities for "
                       "annotation. To add a new label, you can specify a "
                       "patterns file containing examples of the new entity "
                       "as the --patterns argument or pre-train your model "
                       "with examples of the new entity and load it back in.",
                       error=True, exits=1)
    if patterns is None:
        predict = model
        update = model.update
    else:
        matcher = PatternMatcher(model.nlp).from_disk(patterns)
        log("RECIPE: Created PatternMatcher and loaded in patterns", patterns)
        predict, update = combine_models(model, matcher)
    # Split the stream into sentences
    stream = split_sentences(model.orig_nlp, stream)
    # Return components, to construct Controller
    return {
        'view_id': 'ner',
        'dataset': dataset,
        'stream': prefer_uncertain(predict(stream)),
        'update': update,  # callback to update the model in-place
        'exclude': exclude,
        'config': {'lang': model.nlp.lang,
                   'label': (', '.join(label)) if label is not None else 'all'}
    }


@recipe('ner.manual',
        dataset=recipe_args['dataset'],
        spacy_model=recipe_args['spacy_model'],
        source=recipe_args['source'],
        api=recipe_args['api'],
        loader=recipe_args['loader'],
        label=recipe_args['label'],
        exclude=recipe_args['exclude'])
def manual(dataset, spacy_model, source=None, api=None, loader=None,
           label=None, exclude=None):
    """
    Mark spans by token. Requires only a tokenizer and no entity recognizer,
    and doesn't do any active learning.
    """
    log("RECIPE: Starting recipe ner.manual", locals())
    nlp = spacy.load(spacy_model)
    log("RECIPE: Loaded model {}".format(spacy_model))
    labels = get_labels(label, nlp)
    log("RECIPE: Annotating with {} labels".format(len(labels)), labels)
    stream = get_stream(source, api=api, loader=loader, rehash=True,
                        dedup=True, input_key='text')
    stream = add_tokens(nlp, stream)

    return {
        'view_id': 'ner_manual',
        'dataset': dataset,
        'stream': stream,
        'exclude': exclude,
        'config': {'labels': labels}
    }


@recipe('ner.make-gold',
    dataset=recipe_args['dataset'],
    spacy_model=recipe_args['spacy_model'],
    source=recipe_args['source'],
    api=recipe_args['api'],
    loader=recipe_args['loader'],
    label=recipe_args['label'],
    exclude=recipe_args['exclude'])
def make_gold(dataset, spacy_model, source=None, api=None, loader=None,
              exclude=None, label=[]):
    """Create gold data for NER by correcting a model's suggestions."""
    log("RECIPE: Starting recipe ner.make-gold", locals())
    nlp = spacy.load(spacy_model)
    log("RECIPE: Loaded model {}".format(spacy_model))
    labels = get_labels(label, nlp)
    log("RECIPE: Annotating with {} labels".format(len(labels)), labels)

    stream = get_stream(source, api=api, loader=loader, rehash=True,
                        dedup=True, input_key='text')
    stream = split_sentences(nlp, stream)

    def make_tasks(nlp, stream):
        """Add a 'spans' key to each example, with predicted entities."""
        texts = ((eg['text'], eg) for eg in stream)
        for doc, eg in nlp.pipe(texts, as_tuples=True):
            task = {'text': doc.text, 'meta': dict(eg.get('meta', {}))}
     
            task['tokens'] = [{'text': token.text, 'start': token.idx,
                               'end': token.idx + len(token.text), 'id': i}
                               for i, token in enumerate(doc)]
            spans = []
            for ent in doc.ents:
                if labels and ent.label_ not in labels:
                    continue
                spans.append({
                    'token_start': ent.start,
                    'token_end': ent.end-1,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'text': ent.text,
                    'label': ent.label_,
                    'source': nlp.meta.get('name', '?'),
                    'input_hash': eg[INPUT_HASH_ATTR]
                })
            task['spans'] = spans
            task = set_hashes(task)
            yield task

    return {
        'view_id': 'ner_manual',
        'dataset': dataset,
        'stream': make_tasks(nlp, stream),
        'exclude': exclude,
        'update': None,
        'config': {'lang': nlp.lang, 'labels': labels}
    }


@recipe('ner.eval',
        dataset=recipe_args['dataset'],
        model=recipe_args['spacy_model'],
        source=recipe_args['source'],
        api=recipe_args['api'],
        loader=recipe_args['loader'],
        exclude=recipe_args['exclude'],
        whole_text=recipe_args['whole_text'],
        label=recipe_args['entity_label'])
def evaluate(dataset, model, source=None, api=None, loader=None, label=None,
             exclude=None, whole_text=False):
    """
    Evaluate a n NER model and build an evaluation set from a stream.
    """
    log("RECIPE: Starting recipe ner.evaluate", locals())

    model = EntityRecognizer(spacy.load(model), label=label)
    stream = get_stream(source, api=api, loader=loader, rehash=True,
                        input_key='text')
    stream = split_sentences(model.nlp, stream)

    def get_tasks(model, stream):
        tuples = ((eg['text'], eg) for eg in stream)
        for i, (doc, eg) in enumerate(model.nlp.pipe(tuples, as_tuples=True)):
            ents = [(ent.start_char, ent.end_char, ent.label_)
                    for ent in doc.ents]
            if model.labels:
                ents = [seL for seL in ents if seL[2] in model.labels]

            eg['label'] = 'all correct'
            ents = [{'start': s, 'end': e, 'label': L} for s, e, L in ents]
            if whole_text:
                eg['spans'] = ents
                eg = set_hashes(eg, overwrite=True)
                yield eg
            else:
                for span in ents:
                    task = copy.deepcopy(eg)
                    task['spans'] = [span]
                    task = set_hashes(task, overwrite=True)
                    yield task

    return {
        'view_id': 'classification',
        'dataset': dataset,
        'stream': get_tasks(model, stream),
        'exclude': exclude
    }


@recipe('ner.eval-ab',
        dataset=recipe_args['dataset'],
        before_model=recipe_args['spacy_model'],
        after_model=recipe_args['spacy_model'],
        source=recipe_args['source'],
        api=recipe_args['api'],
        loader=recipe_args['loader'],
        label=recipe_args['entity_label'])
def ab_evaluate(dataset, before_model, after_model, source=None, api=None,
                loader=None, label=None, exclude=None):  # pragma: no cover
    """
    Evaluate a n NER model and build an evaluation set from a stream.
    """
    log("RECIPE: Starting recipe ner.eval-ab", locals())

    def get_task(i, text, ents, name):
        spans = [{'start': s, 'end': e, 'label': L} for s, e, L in ents]
        task = {'id': i, 'input': {'text': text},
                'output': {'text': text, 'spans': spans}}
        task[INPUT_HASH_ATTR] = mmh3.hash(name + str(i))
        task[TASK_HASH_ATTR] = mmh3.hash(name + str(i))
        return task

    def get_tasks(model, stream, name):
        tuples = ((eg['text'], eg) for eg in stream)
        for i, (doc, eg) in enumerate(model.nlp.pipe(tuples, as_tuples=True)):
            ents = [(ent.start_char, ent.end_char, ent.label_)
                    for ent in doc.ents]
            if model.labels:
                ents = [seL for seL in ents if seL[2] in model.labels]
            task = get_task(i, eg['text'], ents, name)
            yield task

    before_model = EntityRecognizer(spacy.load(before_model), label=label)
    after_model = EntityRecognizer(spacy.load(after_model), label=label)
    stream = list(get_stream(source, api=api, loader=loader, rehash=True,
                             dedup=True, input_key='text'))
    stream = list(split_sentences(before_model.nlp, stream))
    before_stream = list(get_tasks(before_model, stream, 'before'))
    after_stream = list(get_tasks(after_model, stream, 'after'))
    stream = list(get_compare_questions(before_stream, after_stream, True))

    return {
        'view_id': 'compare',
        'dataset': dataset,
        'stream': stream,
        'on_exit': printers.get_compare_printer('Before', 'After'),
        'exclude': exclude
    }


@recipe('ner.batch-train',
        dataset=recipe_args['dataset'],
        input_model=recipe_args['spacy_model'],
        output_model=recipe_args['output'],
        label=recipe_args['entity_label'],
        factor=recipe_args['factor'],
        dropout=recipe_args['dropout'],
        n_iter=recipe_args['n_iter'],
        batch_size=recipe_args['batch_size'],
        beam_width=recipe_args['beam_width'],
        eval_id=recipe_args['eval_id'],
        eval_split=recipe_args['eval_split'],
        silent=recipe_args['silent'])
def batch_train(dataset, input_model, output_model=None, label='', factor=1,
                dropout=0.2, n_iter=10, batch_size=32, beam_width=16,
                eval_id=None, eval_split=None, silent=False):
    """
    Batch train a Named Entity Recognition model from annotations. Prodigy will
    export the best result to the output directory, and include a JSONL file of
    the training and evaluation examples. You can either supply a dataset ID
    containing the evaluation data, or choose to split off a percentage of
    examples for evaluation.
    """
    log("RECIPE: Starting recipe ner.batch-train", locals())
    print_ = get_print(silent)
    random.seed(0)
    nlp = spacy.load(input_model)
    print_("\nLoaded model {}".format(input_model))
    if 'sentencizer' not in nlp.pipe_names and 'sbd' not in nlp.pipe_names:
        nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)
        log("RECIPE: Added sentence boundary detector to model pipeline",
            nlp.pipe_names)
    examples = merge_spans(DB.get_dataset(dataset))
    random.shuffle(examples)
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        ner.cfg['pretrained_dims'] = 300
        for eg in examples:
            for span in eg.get('spans', []):
                ner.add_label(span['label'])
        for l in label:
            ner.add_label(l)
        nlp.add_pipe(ner, last=True)
        nlp.begin_training()
    else:
        ner = nlp.get_pipe('ner')
        for l in label:
            ner.add_label(l)
    if eval_id:
        evals = DB.get_dataset(eval_id)
        print_("Loaded {} evaluation examples from '{}'"
               .format(len(evals), eval_id))
    else:
        examples, evals, eval_split = split_evals(examples, eval_split)
        print_("Using {}% of accept/reject examples ({}) for evaluation"
               .format(round(eval_split * 100), len(evals)))
    model = EntityRecognizer(nlp, label=label)
    log('RECIPE: Initialised EntityRecognizer with model {}'
        .format(input_model), model.nlp.meta)
    examples = list(split_sentences(model.orig_nlp, examples))
    evals = list(split_sentences(model.orig_nlp, evals))
    baseline = model.evaluate(evals)
    log("RECIPE: Calculated baseline from evaluation examples "
        "(accuracy %.2f)" % baseline['acc'])
    best = None
    random.shuffle(examples)
    examples = examples[:int(len(examples) * factor)]
    print_(printers.trainconf(dropout, n_iter, batch_size, factor,
                              len(examples)))
    print_(printers.ner_before(**baseline))
    if len(evals) > 0:
        print_(printers.ner_update_header())

    for i in range(n_iter):
        losses = model.batch_train(examples, batch_size=batch_size,
                                   drop=dropout, beam_width=beam_width)
        stats = model.evaluate(evals)
        if best is None or stats['acc'] > best[0]:
            model_to_bytes = None
            if output_model is not None:
                model_to_bytes = model.to_bytes()
            best = (stats['acc'], stats, model_to_bytes)
        print_(printers.ner_update(i, losses, stats))
    best_acc, best_stats, best_model = best
    print_(printers.ner_result(best_stats, best_acc, baseline['acc']))
    if output_model is not None:
        model.from_bytes(best_model)
        msg = export_model_data(output_model, model.nlp, examples, evals)
        print_(msg)
    best_stats['baseline'] = baseline['acc']
    best_stats['acc'] = best_acc
    return best_stats


@recipe('ner.train-curve',
        dataset=recipe_args['dataset'],
        input_model=recipe_args['spacy_model'],
        label=recipe_args['entity_label'],
        dropout=recipe_args['dropout'],
        n_iter=recipe_args['n_iter'],
        batch_size=recipe_args['batch_size'],
        beam_width=recipe_args['beam_width'],
        eval_id=recipe_args['eval_id'],
        eval_split=recipe_args['eval_split'],
        n_samples=recipe_args['n_samples'])
def train_curve(dataset, input_model, label='', dropout=0.2, n_iter=5,
                batch_size=32, beam_width=16, eval_id=None, eval_split=None,
                n_samples=4):  # pragma: no cover
    """
    Batch-train models with different portions of the training examples and
    print the accuracy figures and accuracy improvements.
    """
    log("RECIPE: Starting recipe ner.train-curve", locals())
    factors = [(i + 1) / n_samples for i in range(n_samples)]
    prev_acc = 0
    print("\nStarting with model {}".format(input_model))
    print(printers.trainconf(dropout, n_iter, batch_size, samples=n_samples))
    print(printers.ner_curve_header())
    for factor in factors:
        best_stats = batch_train(dataset, input_model=input_model, label=label,
                                 factor=factor, dropout=dropout,
                                 n_iter=n_iter, batch_size=batch_size,
                                 beam_width=beam_width, eval_id=eval_id,
                                 eval_split=eval_split, silent=True)
        print(printers.ner_curve(factor, best_stats, prev_acc))
        prev_acc = best_stats['acc']


@recipe('ner.print-best',
        dataset=recipe_args['dataset'],
        spacy_model=recipe_args['spacy_model'])
def print_best(dataset, spacy_model):
    """
    Predict the highest-scoring parse for examples in a dataset. Scores are
    calculated using the annotations in the dataset, and the statistical model.
    """
    log("RECIPE: Starting recipe ner.best-parse", locals())
    model = EntityRecognizer(spacy.load(spacy_model))
    log('RECIPE: Initialised EntityRecognizer with model {}'
        .format(spacy_model), model.nlp.meta)
    log("RECIPE: Outputting stream of examples")
    for eg in model.make_best(DB.get_dataset(dataset)):
        print(json.dumps(eg))


@recipe('ner.print-stream',
        spacy_model=recipe_args['spacy_model'],
        source=recipe_args['source'],
        api=recipe_args['api'],
        loader=recipe_args['loader'],
        label=recipe_args['entity_label'])
def pretty_print_stream(spacy_model, source=None, api=None, loader=None,
                        label=''):
    """
    Pretty print stream output.
    """
    log("RECIPE: Starting recipe ner.print-stream", locals())

    def add_entities(stream, nlp, labels=None):
        for eg in stream:
            doc = nlp(eg['text'])
            ents = [{'start': e.start_char, 'end': e.end_char,
                     'label': e.label_} for e in doc.ents
                    if not labels or e.label_ in labels]
            if ents:
                eg['spans'] = ents
                yield eg

    nlp = spacy.load(spacy_model)
    stream = get_stream(source, api, loader, rehash=True, input_key='text')
    stream = add_entities(stream, nlp, label)
    printers.pretty_print_ner(stream)


@recipe('ner.print-dataset',
        dataset=recipe_args['dataset'])
def pretty_print_dataset(dataset):  # pragma: no cover
    """
    Pretty print dataset.
    """
    log("RECIPE: Starting recipe ner.print-dataset", locals())
    examples = DB.get_dataset(dataset)
    if not examples:
        raise ValueError("Can't load '{}' from database {}"
                         .format(dataset, DB.db_name))
    printers.pretty_print_ner(examples)
