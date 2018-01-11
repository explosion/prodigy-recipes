# coding: utf8
from __future__ import unicode_literals, print_function

import random
import numpy
import copy
import base64
from cytoolz import partition_all
import lightnet
from lightnet import Image
from lightnet.lightnet import BoxLabels
from prodigy import recipe, recipe_args, log, get_stream, set_hashes
from prodigy.components.preprocess import fetch_images
from prodigy.components.sorters import prefer_uncertain
from prodigy.components.db import connect
from prodigy.components import printers
from prodigy.util import split_evals, export_model_data, INPUT_HASH_ATTR


DB = connect()


@recipe('image.teach',
        dataset=recipe_args['dataset'],
        lightnet_model=("Loadable lightnet model", "positional", None, str),
        source=recipe_args['source'],
        api=recipe_args['api'],
        label=recipe_args['entity_label'],
        exclude=recipe_args['exclude'])
def teach(dataset, lightnet_model, source=None, api=None, label=None,
          exclude=None):
    log("RECIPE: Starting recipe image.teach", locals())
    model = ImageDetector(lightnet.load(lightnet_model), label=label)
    log("RECIPE: Initialised ImageDetector with model {}"
        .format(lightnet_model))
    if label is not None:
        label = [l.lower() for l in label]
        log("RECIPE: Making sure all labels are in the model", label)
        for l in label:
            if not model.has_label(l):
                raise ValueError("Can't find label '{}' in model {}"
                                 .format(l, lightnet_model))
    stream = get_stream(source, api=api, loader='images', input_key='image')
    stream = fetch_images(stream)

    def free_lightnet(controller):
        # This is called as an on_exit hook and works around a current bug in
        # LightNet if we don't dump the model before we exit – probably due to
        # the ordering of how Python is destroying things. I think a
        # __dealloc__ gets called twice and we don't handle it properly? This
        # should be the only ref to the lightnet model, so freeing it here
        # takes care of the problem.
        del model.model
        model.model = None

    return {
        'view_id': 'image',
        'dataset': dataset,
        'stream': prefer_uncertain(model(stream)),
        'exclude': exclude,
        'update': model.update,
        'config': {'label': ', '.join(label) if label is not None else 'all'},
        'on_exit': free_lightnet
    }


@recipe('image.batch-train',
        dataset=recipe_args['dataset'],
        lightnet_model=("Loadable lightnet model", "positional", None, str),
        output_model=recipe_args['output'],
        factor=recipe_args['factor'],
        dropout=recipe_args['dropout'],
        n_iter=recipe_args['n_iter'],
        batch_size=recipe_args['batch_size'],
        eval_id=recipe_args['eval_id'],
        eval_split=recipe_args['eval_split'],
        silent=recipe_args['silent'])
def batch_train(dataset, lightnet_model, output_model=None, factor=1,
                dropout=0.0, n_iter=20, batch_size=128, eval_id=None,
                eval_split=None,  silent=False):
    """
    Batch train an image model from annotations. Prodigy will export the best
    result to the output directory, and include a JSONL file of the training
    and evaluation examples. You can either supply a dataset ID containing the
    evaluation data, or choose to split off a percentage of examples for
    evaluation.
    """
    log("RECIPE: Starting recipe image.batch-train", locals())

    def get_print(silent):
        if silent:
            return lambda string: None
        else:
            return print

    print_ = get_print(silent)
    model = lightnet.load(lightnet_model)
    log("RECIPE: Loaded model {}".format(lightnet_model))
    examples = DB.get_dataset(dataset)
    random.shuffle(examples)
    if eval_id:
        evals = DB.get_dataset(eval_id)
        print_("Loaded {} evaluation examples from '{}'"
               .format(len(evals), eval_id))
    else:
        examples, evals, eval_split = split_evals(examples, eval_split)
        print_("Using {}% of examples ({}) for evaluation"
               .format(round(eval_split * 100), len(evals)))
    baseline = evaluate(model, evals)
    log("RECIPE: Calculated baseline from evaluation examples "
        "(accuracy %.4f)" % baseline['acc'])
    best = None
    random.shuffle(examples)
    examples = examples[:int(len(examples) * factor)]
    print_(printers.trainconf(dropout, n_iter, batch_size, factor,
                              len(examples)))
    print_(printers.image_before(**baseline))
    if len(evals) > 0:
        print_(printers.image_update_header())
    for i in range(n_iter):
        losses = train_lightnet(model, examples, batch_size=batch_size,
                                dropout=dropout)
        stats = evaluate(model, evals)
        if best is None or stats['acc'] > best[0]:
            best = (stats['acc'], stats, model.to_bytes())
        print_(printers.image_update(i, losses, stats))
    best_acc, best_stats, best_model = best
    print_(printers.image_result(best_stats, best_acc, baseline['acc']))
    if output_model is not None:
        model.model.from_bytes(best_model)
        msg = export_model_data(output_model, model.model, examples, evals)
        print_(msg)
    best_stats['baseline'] = baseline['acc']
    best_stats['acc'] = best_acc
    return best_stats


@recipe('image.train-curve',
        dataset=recipe_args['dataset'],
        lightnet_model=("Loadable lightnet model", "positional", None, str),
        dropout=recipe_args['dropout'],
        n_iter=recipe_args['n_iter'],
        batch_size=recipe_args['batch_size'],
        eval_id=recipe_args['eval_id'],
        eval_split=recipe_args['eval_split'],
        n_samples=recipe_args['n_samples'])
def train_curve(dataset, lightnet_model, dropout=0.0, n_iter=20,
                batch_size=128, eval_id=None, eval_split=None, n_samples=4):
    """
    Batch-train models with different portions of the training examples and
    print the accuracy figures and accuracy improvements.
    """
    log("RECIPE: Starting recipe image.train-curve", locals())
    factors = [(i + 1) / n_samples for i in range(n_samples)]
    prev_acc = 0
    print("\nStarting with model {}".format(lightnet_model))
    print(printers.trainconf(dropout, n_iter, batch_size, samples=n_samples))
    print(printers.image_curve_header())
    for factor in factors:
        best_stats = batch_train(dataset, lightnet_model=lightnet_model,
                                 factor=factor, dropout=dropout,
                                 n_iter=n_iter, batch_size=batch_size,
                                 eval_id=eval_id, eval_split=eval_split,
                                 silent=True)
        print(printers.image_curve(factor, best_stats, prev_acc))
        prev_acc = best_stats['acc']


class ImageDetector(object):
    model_type = 'image-detector'

    def __init__(self, model, label=None):
        self.model = model
        self.labels = label

    def has_label(self, label):
        return label in self.model.names

    def __call__(self, stream, thresh=0.5):
        log("MODEL: Getting predictions for images")
        if self.labels:
            log("MODEL: Only asking for boxes labelled %s" % self.labels)
        for eg in stream:
            if not eg['image'].startswith('data:'):
                invalid = eg['image']
                msg = ("Invalid image: Expected base64-encoded data URI, but "
                       "got: '{}'. If you're using a custom recipe, you can "
                       "call prodigy.components.preprocess.fetch_images on "
                       "your stream to ensure all images are converted.")
                raise ValueError(msg.format(invalid if len(invalid) < 100
                                            else invalid[:100] + '...'))
            image = Image.from_bytes(b64_uri_to_bytes(eg['image']))
            boxes = self.model(image, thresh=thresh)
            boxes = [box for box in boxes if box[2] >= thresh]
            eg['width'] = image.width
            eg['height'] = image.height
            eg['spans'] = [get_span(box) for box in boxes]
            for i in range(len(eg['spans'])):
                if self.labels and eg['spans'][i]['label'] not in self.labels:
                    continue
                task = copy.deepcopy(eg)
                task['spans'][i]['hidden'] = False
                task = set_hashes(task, overwrite=True)
                score = task['spans'][i]['score']
                task['score'] = score
                yield score, task
        log("MODEL: Exiting ImageModel()")

    def update(self, examples):
        answers = {}
        for eg in examples:
            key = eg[INPUT_HASH_ATTR]
            for i, span in enumerate(eg['spans']):
                if not span['hidden'] and eg['answer'] != 'ignore':
                    answers[(key, i)] = eg['answer']
        Xs = []
        ys = []
        seen = set()
        for eg in examples:
            key = eg[INPUT_HASH_ATTR]
            if key in seen:
                continue
            seen.add(key)
            image = Image.from_bytes(b64_uri_to_bytes(eg['image']))
            ids = []
            boxes = []
            for i, span in enumerate(eg['spans']):
                if answers.get((key, i)) == 'reject':
                    continue
                w, h = get_size(span['points'])
                x, y = span['center']
                rel_points = abs2rel(eg['width'], eg['height'], [x, y, w, h])
                ids.append(span['label_id'])
                boxes.append(rel_points)
            if ids:
                Xs.append(image)
                ys.append(BoxLabels(numpy.asarray(ids, dtype='i'),
                                    numpy.asarray(boxes, dtype='f')))
        if Xs:
            loss = self.model.update(Xs, ys)
        else:
            loss = 0
        log('MODEL: batch loss = %s' % loss)
        return loss


def train_lightnet(model, examples, *, batch_size=16, dropout=0.0):
    if dropout != 0.0:
        raise NotImplementedError
    detector = ImageDetector(model)
    loss = 0.
    random.shuffle(examples)
    for batch in partition_all(batch_size, examples):
        loss += detector.update(batch)
    return {'detection': loss}


def evaluate(model, evals, thresh=0.5):
    stats = {'right': 0., 'wrong': 0., 'objects': 0.}
    for eg in evals:
        if eg['answer'] == 'ignore':
            continue
        image = Image.from_bytes(b64_uri_to_bytes(eg['image']))
        guesses = BoxLabels.from_results(model(image))
        truths = _get_truths(eg)
        for box in truths:
            if guesses.has_box(box):
                if eg['answer'] == 'accept':
                    stats['right'] += 1
                else:
                    stats['wrong'] += 1
        stats['objects'] += len(guesses)
    stats['acc'] = stats['right'] / (stats['right'] + stats['wrong'])
    log("MODEL: Evaluated {} examples".format(len(evals)), stats)
    return stats


def _get_truths(eg):
    output = []
    for box in eg['spans']:
        if box['hidden']:
            continue
        w, h = get_size(box['points'])
        x, y = box['center']
        id_ = box['label_id']
        name = box['label']
        output.append({'id': id_, 'x': x, 'y': y, 'w': w, 'h': h,
                       'name': name})
    return output


def get_span(box, hidden=True):
    class_id, name, prob, abs_points = box
    if not isinstance(name, str):
        name = str(name, 'utf8')
    rel_points = get_points(abs_points)
    return {'score': prob,
            'label': name,
            'label_id': class_id,
            'points': rel_points,
            'center': [abs_points[0], abs_points[1]],
            'hidden': hidden}


def get_points(points):
    """Get (x, y) coordinates of all four bounding box edges from
    (x, y, w, h) tuples.
    """
    x, y, w, h = points
    return [(x - w/2, y - h/2), (x - w/2, y + h/2), (x + w/2, y + h/2),
            (x + w/2, y - h/2)]


def get_size(points):
    """Get image width and height from list of (x, y) coordinates describing
    the bounding box edges.
    """
    (bl_x, bl_y), (tl_x, tl_y), (tr_x, tr_y), (br_x, br_y) = points
    width = tr_x - tl_x
    height = tl_y - bl_y
    return (width, height)


def abs2rel(image_w, image_h, xywh):
    """Convert absolute to relative coordinates with respect to image size."""
    x, y, w, h = xywh
    rel_x = x / image_w
    rel_y = y / image_h
    rel_w = w / image_w
    rel_h = h / image_h
    return rel_x, rel_y, rel_w, rel_h


def b64_uri_to_bytes(data_uri):
    data = data_uri.split('base64,', 1)[1]
    return base64.decodestring(bytes(data, 'ascii'))
