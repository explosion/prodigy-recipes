# coding: utf8
from __future__ import unicode_literals

import pytest
import tempfile
from contextlib import contextmanager
from prodigy.components.db import connect
from prodigy.util import write_jsonl, INPUT_HASH_ATTR, TASK_HASH_ATTR
from prodigy.models.ner import merge_spans

from ner.ner_teach import ner_teach
from ner.ner_match import ner_match
from ner.ner_manual import ner_manual
from ner.ner_make_gold import ner_make_gold
from ner.ner_silver_to_gold import ner_silver_to_gold
from textcat.textcat_teach import textcat_teach
from textcat.textcat_custom_model import textcat_custom_model
from terms.terms_teach import terms_teach
from image.image_manual import image_manual
from other.mark import mark
from other.choice import choice


@pytest.fixture()
def dataset():
    return False


@pytest.fixture
def spacy_model():
    return 'en_core_web_sm'


@pytest.fixture
def vectors():
    return 'en_core_web_md'


@pytest.fixture
def labels():
    return ['PERSON', 'ORG']


@pytest.fixture()
def source():
    texts = ['This is a text about David Bowie', 'Apple makes iPhones']
    examples = [{'text': text} for text in texts]
    _, tmp_file = tempfile.mkstemp()
    write_jsonl(tmp_file, examples)
    return tmp_file


@pytest.fixture()
def patterns():
    examples = [{'label': 'PERSON', 'pattern': 'David Bowie'},
                {'label': 'ORG', 'pattern': [{'lower': 'apple'}]}]
    _, tmp_file = tempfile.mkstemp()
    write_jsonl(tmp_file, examples)
    return tmp_file


@contextmanager
def tmp_dataset(name, examples=[]):
    DB = connect()
    DB.add_dataset(name)
    DB.add_examples(examples, datasets=[name])
    yield examples
    DB.drop_dataset(name)


def test_ner_teach(dataset, spacy_model, source, labels, patterns):
    recipe = ner_teach(dataset, spacy_model, source, labels, patterns)
    stream = list(recipe['stream'])
    assert recipe['view_id'] == 'ner'
    assert recipe['dataset'] == dataset
    assert len(stream) == 5
    assert 'spans' in stream[0]
    assert 'tokens' in stream[0]
    assert 'meta' in stream[0]
    assert 'score' in stream[0]['meta']


def test_ner_match(dataset, spacy_model, source, patterns):
    recipe = ner_match(dataset, spacy_model, source, patterns)
    stream = list(recipe['stream'])
    assert recipe['view_id'] == 'ner'
    assert recipe['dataset'] == dataset
    assert len(stream) == 2
    assert 'spans' in stream[0]
    assert len(stream[0]['spans']) == 1
    assert stream[0]['spans'][0]['label'] == 'PERSON'
    assert 'spans' in stream[1]
    assert len(stream[1]['spans']) == 1
    assert stream[1]['spans'][0]['label'] == 'ORG'


def test_ner_manual(dataset, spacy_model, source, labels):
    recipe = ner_manual(dataset, spacy_model, source, labels)
    stream = list(recipe['stream'])
    assert recipe['view_id'] == 'ner_manual'
    assert recipe['dataset'] == dataset
    assert len(stream) == 2
    assert 'tokens' in stream[0]
    assert 'tokens' in stream[1]


def test_ner_make_gold(dataset, spacy_model, source, labels):
    recipe = ner_make_gold(dataset, spacy_model, source, labels)
    stream = list(recipe['stream'])
    assert recipe['view_id'] == 'ner_manual'
    assert recipe['dataset'] == dataset
    assert len(stream) == 2
    assert 'spans' in stream[0]
    assert 'tokens' in stream[0]


def test_ner_silver_to_gold(dataset, spacy_model):
    silver_dataset = '__test_ner_silver_to_gold__'
    silver_examples = [
        {
            INPUT_HASH_ATTR: 1,
            TASK_HASH_ATTR: 11,
            'text': 'Hello world',
            'answer': 'accept',
            'spans': [{'start': 0, 'end': 5, 'label': 'PERSON'}]
        },
        {
            INPUT_HASH_ATTR: 1,
            TASK_HASH_ATTR: 12,
            'text': 'Hello world',
            'answer': 'reject',
            'spans': [{'start': 6, 'end': 11, 'label': 'PERSON'}]
        },
        {
            INPUT_HASH_ATTR: 2,
            TASK_HASH_ATTR: 21,
            'text': 'This is a test',
            'answer': 'reject',
            'spans': [{'start': 5, 'end': 7, 'label': 'ORG'}]
        }
    ]
    with tmp_dataset(silver_dataset, silver_examples):
        recipe = ner_silver_to_gold(silver_dataset, dataset, spacy_model)
        stream = list(recipe['stream'])
    assert recipe['view_id'] == 'ner_manual'
    assert recipe['dataset'] == dataset
    assert len(stream) == 2
    assert stream[0]['text'] == 'Hello world'
    assert 'tokens' in stream[0]
    assert stream[1]['text'] == 'This is a test'
    assert 'tokens' in stream[1]


def test_textcat_teach(dataset, spacy_model, source, labels, patterns):
    recipe = textcat_teach(dataset, spacy_model, source, labels, patterns)
    stream = list(recipe['stream'])
    assert recipe['view_id'] == 'classification'
    assert recipe['dataset'] == dataset
    assert len(stream) >= 2
    assert 'label' in stream[0]
    assert 'meta' in stream[0]
    assert 'score' in stream[0]['meta']


def test_textcat_custom_model(dataset, source, labels):
    recipe = textcat_custom_model(dataset, source, labels)
    stream = list(recipe['stream'])
    assert recipe['view_id'] == 'classification'
    assert recipe['dataset'] == dataset
    assert len(stream) >= 1
    assert 'label' in stream[0]


def test_terms_teach(dataset, vectors):
    seeds = ['cat', 'dog', 'mouse']
    recipe = terms_teach(dataset, vectors, seeds)
    assert recipe['view_id'] == 'text'
    assert recipe['dataset'] == dataset


def test_image_manual(dataset):
    img_dir = tempfile.mkdtemp()
    img1 = tempfile.NamedTemporaryFile(dir=img_dir, prefix='1', suffix='.jpg')
    img2 = tempfile.NamedTemporaryFile(dir=img_dir, prefix='2', suffix='.png')
    no_img = tempfile.NamedTemporaryFile(dir=img_dir, prefix='3', suffix='.txt')
    recipe = image_manual(dataset, img_dir, ['PERSON', 'DOG', 'CAT'])
    stream = list(recipe['stream'])
    assert recipe['view_id'] == 'image_manual'
    assert recipe['dataset'] == dataset
    assert len(stream) == 2


def test_mark(dataset, source):
    view_id = 'text'
    recipe = mark(dataset, source, view_id)
    stream = list(recipe['stream'])
    assert recipe['view_id'] == view_id
    assert recipe['dataset'] == dataset
    assert len(stream) == 2
    assert hasattr(recipe['update'], '__call__')
    assert hasattr(recipe['on_load'], '__call__')
    assert hasattr(recipe['on_exit'], '__call__')


def test_choice(dataset, source):
    options = ['OPTION_A', 'OPTION_B', 'OPTION_C']
    recipe = choice(dataset, source, options)
    stream = list(recipe['stream'])
    assert recipe['view_id'] == 'choice'
    assert recipe['dataset'] == dataset
    assert len(stream) == 2
    assert 'options' in stream[0]
    assert len(stream[0]['options']) == 3
    assert stream[0]['options'][0]['id'] == 'OPTION_A'
    assert recipe['config']['choice_style'] == 'single'
    assert recipe['config']['choice_auto_accept']
