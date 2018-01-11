# coding: utf8
from __future__ import unicode_literals, print_function

import copy
from collections import Counter

from ..components import printers
from ..components.loaders import get_stream
from ..components.preprocess import fetch_images
from ..core import recipe, recipe_args
from ..util import TASK_HASH_ATTR, set_hashes, prints, log, b64_uri_to_bytes


@recipe('mark',
        dataset=recipe_args['dataset'],
        source=recipe_args['source'],
        api=recipe_args['api'],
        loader=recipe_args['loader'],
        label=recipe_args['label'],
        view_id=recipe_args['view'],
        memorize=recipe_args['memorize'],
        exclude=recipe_args['exclude'])
def mark(dataset, source=None, view_id=None, label='', api=None,
         loader=None, memorize=False, exclude=None):
    """
    Click through pre-prepared examples, with no model in the loop.
    """
    log('RECIPE: Starting recipe mark', locals())
    stream = get_stream(source, api, loader)
    counts = Counter()
    memory = {}

    def fill_memory(ctrl):
        if memorize:
            examples = ctrl.db.get_dataset(dataset)
            log("RECIPE: Add {} examples from dataset '{}' to memory"
                .format(len(examples), dataset))
            for eg in examples:
                memory[eg[TASK_HASH_ATTR]] = eg['answer']

    def ask_questions(stream):
        for eg in stream:
            if TASK_HASH_ATTR in eg and eg[TASK_HASH_ATTR] in memory:
                answer = memory[eg[TASK_HASH_ATTR]]
                counts[answer] += 1
            else:
                if label:
                    eg['label'] = label
                yield eg

    def recv_answers(answers):
        for eg in answers:
            counts[eg['answer']] += 1
            memory[eg[TASK_HASH_ATTR]] = eg['answer']

    def print_results(ctrl):
        print(printers.answers(counts))

    return {
        'view_id': view_id,
        'dataset': dataset,
        'stream': ask_questions(stream),
        'exclude': exclude,
        'update': recv_answers,
        'on_load': fill_memory,
        'on_exit': print_results,
        'config': {'label': label}
    }


@recipe('image.test',
        dataset=recipe_args['dataset'],
        lightnet_model=("Loadable lightnet model", "positional", None, str),
        source=recipe_args['source'],
        api=recipe_args['api'],
        exclude=recipe_args['exclude'])
def image_test(dataset, lightnet_model, source=None, api=None, exclude=None):
    """
    Test Prodigy's image annotation interface with a YOLOv2 model loaded
    via LightNet. Requires the LightNet library to be installed. The recipe
    will find objects in the images, and create a task for each object.
    """
    log("RECIPE: Starting recipe image.test", locals())
    try:
        import lightnet
    except ImportError:
        prints("Can't find LightNet", "In order to use this recipe, you "
               "need to have LightNet installed (currently compatible with "
               "Mac and Linux): pip install lightnet. For more details, see: "
               "https://github.com/explosion/lightnet", error=True, exits=1)

    def get_image_stream(model, stream, thresh=0.5):
        for eg in stream:
            if not eg['image'].startswith('data'):
                msg = "Expected base64-encoded data URI, but got: '{}'."
                raise ValueError(msg.format(eg['image'][:100]))
            image = lightnet.Image.from_bytes(b64_uri_to_bytes(eg['image']))
            boxes = [b for b in model(image, thresh=thresh) if b[2] >= thresh]
            eg['width'] = image.width
            eg['height'] = image.height
            eg['spans'] = [get_span(box) for box in boxes]
            for i in range(len(eg['spans'])):
                task = copy.deepcopy(eg)
                task['spans'][i]['hidden'] = False
                task = set_hashes(task, overwrite=True)
                score = task['spans'][i]['score']
                task['score'] = score
                yield task

    def get_span(box, hidden=True):
        class_id, name, prob, abs_points = box
        name = str(name, 'utf8') if not isinstance(name, str) else name
        x, y, w, h = abs_points
        rel_points = [(x - w/2, y - h/2), (x - w/2, y + h/2),
                      (x + w/2, y + h/2), (x + w/2, y - h/2)]
        return {'score': prob, 'label': name, 'label_id': class_id,
                'points': rel_points, 'center': [abs_points[0], abs_points[1]],
                'hidden': hidden}

    model = lightnet.load(lightnet_model)
    log("RECIPE: Loaded LightNet model {}".format(lightnet_model))
    stream = get_stream(source, api=api, loader='images', input_key='image')
    stream = fetch_images(stream)

    def free_lighnet(ctrl):
        nonlocal model
        del model

    return {
        'view_id': 'image',
        'dataset': dataset,
        'stream': get_image_stream(model, stream),
        'exclude': exclude,
        'on_exit': free_lighnet
    }
