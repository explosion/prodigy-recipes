# coding: utf8
from __future__ import unicode_literals, print_function

import spacy
import random
import cytoolz
import tqdm

from prodigy.models.textcat import TextClassifier
from prodigy.components import printers
from prodigy.components.loaders import get_stream
from prodigy.components.preprocess import split_sentences
from prodigy.components.db import connect
from prodigy.components.sorters import prefer_uncertain, find_with_terms
from prodigy.core import recipe, recipe_args
from prodigy.util import export_model_data, split_evals, get_print
from prodigy.util import get_seeds, get_seeds_from_set, log


DB = connect()


@recipe('textcat.teach',
        dataset=recipe_args['dataset'],
        spacy_model=recipe_args['spacy_model'],
        source=recipe_args['source'],
        label=recipe_args['label'],
        api=recipe_args['api'],
        loader=recipe_args['loader'],
        seeds=recipe_args['seeds'],
        long_text=("Long text", "flag", "L", bool),
        exclude=recipe_args['exclude'])
def teach(dataset, spacy_model, source=None, label='', api=None,
          loader=None, seeds=None, long_text=False, exclude=None):
    """
    Collect the best possible training data for a text classification model
    with the model in the loop. Based on your annotations, Prodigy will decide
    which questions to ask next.
    """
    log('RECIPE: Starting recipe textcat.teach', locals())
    nlp = spacy.load(spacy_model)
    log('RECIPE: Creating TextClassifier with model {}'
        .format(spacy_model))
    model = TextClassifier(nlp, label.split(','), long_text=long_text)
    stream = get_stream(source, api, loader, rehash=True, dedup=True,
                        input_key='text')
    if seeds is not None:
        if isinstance(seeds, str) and seeds in DB:
            seeds = get_seeds_from_set(seeds, DB.get_dataset(seeds))
        else:
            seeds = get_seeds(seeds)
        # Find 'seedy' examples
        examples_with_seeds = list(find_with_terms(stream, seeds,
                                   at_least=10, at_most=1000,
                                   give_up_after=10000))
        for eg in examples_with_seeds:
            eg.setdefault('meta', {})
            eg['meta']['via_seed'] = True
        print("Found {} examples with seeds".format(len(examples_with_seeds)))
        examples_with_seeds = [task for _, task in model(examples_with_seeds)]
    # Rank the stream. Note this is continuous, as model() is a generator.
    # As we call model.update(), the ranking of examples changes.
    stream = prefer_uncertain(model(stream))
    # Prepend 'seedy' examples, if present
    if seeds:
        log("RECIPE: Prepending examples with seeds to the stream")
        stream = cytoolz.concat((examples_with_seeds, stream))
    return {
        'view_id': 'classification',
        'dataset': dataset,
        'stream': stream,
        'exclude': exclude,
        'update': model.update,
        'config': {'lang': nlp.lang, 'labels': model.labels}
    }


@recipe('textcat.batch-train',
        dataset=recipe_args['dataset'],
        input_model=recipe_args['spacy_model'],
        output_model=recipe_args['output'],
        lang=recipe_args['lang'],
        factor=recipe_args['factor'],
        dropout=recipe_args['dropout'],
        n_iter=recipe_args['n_iter'],
        batch_size=recipe_args['batch_size'],
        eval_id=recipe_args['eval_id'],
        eval_split=recipe_args['eval_split'],
        long_text=("Long text", "flag", "L", bool),
        silent=recipe_args['silent'])
def batch_train(dataset, input_model=None, output_model=None, lang='en',
                factor=1, dropout=0.2, n_iter=10, batch_size=10,
                eval_id=None, eval_split=None, long_text=False, silent=False):
    """
    Batch train a new text classification model from annotations. Prodigy will
    export the best result to the output directory, and include a JSONL file of
    the training and evaluation examples. You can either supply a dataset ID
    containing the evaluation data, or choose to split off a percentage of
    examples for evaluation.
    """
    log("RECIPE: Starting recipe textcat.batch-train", locals())
    print_ = get_print(silent)
    random.seed(0)
    if input_model is not None:
        nlp = spacy.load(input_model, disable=['ner'])
        print_('\nLoaded model {}'.format(input_model))
    else:
        nlp = spacy.blank(lang, pipeline=[])
        print_('\nLoaded blank model')
    examples = DB.get_dataset(dataset)
    labels = {eg['label'] for eg in examples}
    labels = list(sorted(labels))
    model = TextClassifier(nlp, labels, long_text=long_text,
                           low_data=len(examples) < 1000)
    log('RECIPE: Initialised TextClassifier with model {}'
        .format(input_model), model.nlp.meta)
    random.shuffle(examples)
    if eval_id:
        evals = DB.get_dataset(eval_id)
        print_("Loaded {} evaluation examples from '{}'"
               .format(len(evals), eval_id))
    else:
        examples, evals, eval_split = split_evals(examples, eval_split)
        print_("Using {}% of examples ({}) for evaluation"
               .format(round(eval_split * 100), len(evals)))
    random.shuffle(examples)
    examples = examples[:int(len(examples) * factor)]
    print_(printers.trainconf(dropout, n_iter, batch_size, factor,
                              len(examples)))
    if len(evals) > 0:
        print_(printers.tc_update_header())
    best_acc = {'accuracy': 0}
    best_model = None
    if long_text:
        examples = list(split_sentences(nlp, examples))
    for i in range(n_iter):
        loss = 0.
        random.shuffle(examples)
        for batch in cytoolz.partition_all(batch_size,
                                           tqdm.tqdm(examples, leave=False)):
            batch = list(batch)
            loss += model.update(batch, revise=False, drop=dropout)
        if len(evals) > 0:
            with nlp.use_params(model.optimizer.averages):
                acc = model.evaluate(tqdm.tqdm(evals, leave=False))
                if acc['accuracy'] > best_acc['accuracy']:
                    best_acc = dict(acc)
                    best_model = nlp.to_bytes()
            print_(printers.tc_update(i, loss, acc))
    if len(evals) > 0:
        print_(printers.tc_result(best_acc))
    if output_model is not None:
        if best_model is not None:
            nlp = nlp.from_bytes(best_model)
        msg = export_model_data(output_model, nlp, examples, evals)
        print_(msg)
    return best_acc['accuracy']


@recipe('textcat.train-curve',
        dataset=recipe_args['dataset'],
        input_model=recipe_args['spacy_model'],
        dropout=recipe_args['dropout'],
        n_iter=recipe_args['n_iter'],
        batch_size=recipe_args['batch_size'],
        eval_id=recipe_args['eval_id'],
        eval_split=recipe_args['eval_split'],
        n_samples=recipe_args['n_samples'])
def train_curve(dataset, input_model=None, dropout=0.2, n_iter=5,
                batch_size=10, eval_id=None, eval_split=None, n_samples=4):
    """
    Batch-train models with different portions of the training examples and
    print the accuracy figures and accuracy improvements.
    """
    log("RECIPE: Starting recipe textcat.train-curve", locals())
    factors = [(i + 1) / n_samples for i in range(n_samples)]
    prev_acc = 0
    if input_model is not None:
        print("\nStarting with model {}".format(input_model))
    else:
        print("\nStarting with blank model")
    print(printers.trainconf(dropout, n_iter, batch_size, samples=n_samples))
    print(printers.tc_curve_header())
    for factor in factors:
        best_acc = batch_train(dataset, input_model=input_model,
                               factor=factor, dropout=dropout, n_iter=n_iter,
                               batch_size=batch_size, eval_id=eval_id,
                               eval_split=eval_split, silent=True)
        print(printers.tc_curve(factor, best_acc, prev_acc))
        prev_acc = best_acc


@recipe('textcat.eval',
        dataset=recipe_args['dataset'],
        spacy_model=recipe_args['spacy_model'],
        source=recipe_args['source'],
        label=recipe_args['label'],
        api=recipe_args['api'],
        loader=recipe_args['loader'],
        exclude=recipe_args['exclude'])
def evaluate(dataset, spacy_model, source, label='', api=None,
             loader=None, exclude=None):
    """
    Evaluate a text classification model and build an evaluation set from a
    stream.
    """
    log("RECIPE: Starting recipe textcat.eval", locals())
    nlp = spacy.load(spacy_model, disable=['tagger', 'parser', 'ner'])
    stream = get_stream(source, api, loader)
    model = TextClassifier(nlp, label)
    log('RECIPE: Initialised TextClassifier with model {}'
        .format(spacy_model), model.nlp.meta)

    def on_exit(ctrl):
        examples = ctrl.db.get_dataset(dataset)
        data = dict(model.evaluate(examples))
        print(printers.tc_result(data))

    return {
        'view_id': 'classification',
        'dataset': dataset,
        'stream': stream,
        'exclude': exclude,
        'on_exit': on_exit,
        'config': {'lang': nlp.lang, 'labels': model.labels}
    }


@recipe('textcat.print-stream',
        source=recipe_args['source'],
        label=recipe_args['label'],
        api=recipe_args['api'],
        loader=recipe_args['loader'])
def pretty_print_stream(source, label='', api=None, loader=None):
    """
    Pretty print stream output.
    """
    log("RECIPE: Starting recipe textcat.print-stream", locals())
    stream = get_stream(source, api, loader)
    printers.pretty_print_tc(stream, label)


@recipe('textcat.print-dataset',
        dataset=recipe_args['dataset'])
def pretty_print_dataset(dataset):
    """
    Pretty print dataset.
    """
    log("RECIPE: Starting recipe textcat.print-dataset", locals())
    examples = DB.get_dataset(dataset)
    if not examples:
        raise ValueError("Can't load '{}' from database {}"
                         .format(dataset, DB.db_name))
    printers.pretty_print_tc(examples)
