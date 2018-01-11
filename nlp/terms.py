# coding: utf8
from __future__ import unicode_literals

import sys
import spacy
import json
from spacy.tokens import Doc

from ..core import recipe, recipe_args
from ..components.db import connect
from ..components.loaders import get_stream
from ..components import preprocess
from ..components.sorters import Probability
from ..util import get_word2vec, prints, get_seeds, set_hashes
from ..util import write_jsonl, log


DB = connect()


@recipe('terms.train-vectors',
        output_model=recipe_args['output_model'],
        source=recipe_args['source_file'],
        loader=recipe_args['loader'],
        spacy_model=("Loadable spaCy model", "option", "sm", str),
        lang=recipe_args['lang'],
        size=("Dimension of the word vectors", "option", "d", int),
        window=("Context window size", "option", "w", int),
        min_count=("Min count", "option", "m", int),
        negative=("Number of negative samples", "option", "g", int),
        n_iter=recipe_args['n_iter'],
        n_workers=("Number of workers", "option", "nw", int),
        merge_ents=("Merge named entities", "flag", "ME", bool),
        merge_nps=("Merge noun phrases", "flag", "MN", bool))
def train_vectors(output_model, source=None, loader=None, spacy_model=None,
                  lang='xx', size=128, window=5, min_count=10, negative=5,
                  n_iter=2, n_workers=4, merge_ents=False, merge_nps=False):
    """Train word vectors from a text source."""
    log("RECIPE: Starting recipe terms.train-vectors", locals())
    if spacy_model is None:
        nlp = spacy.blank(lang)
        print("Using blank spaCy model ({})".format(lang))
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        log("RECIPE: Added sentence boundary detector to blank model")
    else:
        nlp = spacy.load(spacy_model)
    if merge_ents:
        nlp.add_pipe(preprocess.merge_entities, name='merge_entities')
        log("RECIPE: Added pipeline component to merge entities")
    if merge_nps:
        nlp.add_pipe(preprocess.merge_noun_chunks, name='merge_noun_chunks')
        log("RECIPE: Added pipeline component to merge noun chunks")
    Word2Vec = get_word2vec()
    if not output_model.exists():
        output_model.mkdir(parents=True)
        log("RECIPE: Created output directory")
    stream = get_stream(source, loader=loader, input_key='text')
    sentences = []
    for doc in nlp.pipe((eg['text'] for eg in stream)):
        for sent in doc.sents:
            sentences.append([w.text for w in doc])
    print("Extracted {} sentences".format(len(sentences)))
    w2v = Word2Vec(sentences, size=size, window=window, min_count=min_count,
                   sample=1e-5, iter=n_iter, workers=n_workers,
                   negative=negative)
    log("RECIPE: Resetting vectors with size {}".format(size))
    nlp.vocab.reset_vectors(width=size)
    log("RECIPE: Adding {} vectors to model vocab".format(len(w2v.wv.vocab)))
    for word in w2v.wv.vocab:
        nlp.vocab.set_vector(word, w2v.wv.word_vec(word))
    nlp.to_disk(output_model)
    prints('Trained Word2Vec model', output_model.resolve())
    return False


@recipe('terms.teach',
        dataset=recipe_args['dataset'],
        vectors=("Loadable spaCy model or path to word2vec file"),
        seeds=recipe_args['seeds'])
def teach(dataset, vectors, seeds=None):
    """
    Bootstrap Prodigy with word vectors and seeds. Seeds can either be a
    path to a newline-separated text file, or a string with comma-separated
    terms.
    """
    log("RECIPE: Starting recipe terms.teach", locals())
    seeds = get_seeds(seeds)
    seed_tasks = [set_hashes({'text': s, 'answer': 'accept'}) for s in seeds]
    DB.add_examples(seed_tasks)
    nlp = spacy.load(vectors)
    log("RECIPE: Loaded vectors from {}".format(vectors))
    accept_doc = Doc(nlp.vocab, words=seeds)
    reject_doc = Doc(nlp.vocab)
    score = 0

    def predict(term):
        nonlocal accept_doc, reject_doc
        if len(accept_doc) == 0 and len(reject_doc) == 0:
            return 0.5
        accept_score = max(term.similarity(accept_doc), 0.0)
        reject_score = max(term.similarity(reject_doc), 0.0)
        score = accept_score / (accept_score + reject_score + 0.2)
        return max(score, 0.0)

    def update(answers):
        nonlocal accept_doc, reject_doc, score
        log("RECIPE: Update predictions with {} answers".format(len(answers)),
            answers)
        accept_words = [t.text for t in accept_doc]
        reject_words = [t.text for t in reject_doc]
        for answer in answers:
            if answer['answer'] == 'accept':
                score += 1
                accept_words.append(answer['text'])
            elif answer['answer'] == 'reject':
                score -= 1
                reject_words.append(answer['text'])
        accept_doc = Doc(nlp.vocab, words=accept_words)
        reject_doc = Doc(nlp.vocab, words=reject_words)

    def stream_scored(stream):
        lexemes = [lex for lex in nlp.vocab if lex.is_alpha and lex.is_lower]
        while True:
            seen = set(w.orth for w in accept_doc)
            seen.update(set(w.orth for w in reject_doc))
            lexemes = [w for w in lexemes if w.orth not in seen]
            by_score = [(predict(lex), lex) for lex in lexemes]
            by_score.sort(reverse=True)
            for _, term in by_score:
                # Need to predict in loop, as model changes
                score = predict(term)
                yield score, {'text': term.text, 'meta': {'score': score}}

    return {
        'dataset': dataset,
        'view_id': 'text',
        'stream': Probability(stream_scored(nlp.vocab)),
        'update': update
    }


@recipe('terms.to-patterns',
        dataset=recipe_args['dataset'],
        output_file=recipe_args['output_file'],
        label=recipe_args['label'])
def to_patterns(dataset=None, label=None, output_file=None):
    """
    Convert a list of seed terms to a list of match patterns that can be used
    with ner.match. If no output file is specified, each pattern is printed
    so the recipe's output can be piped forward to ner.match.
    """
    def get_pattern(term, label):
        return {'label': label, 'pattern': [{'lower': term['text']}]}

    log("RECIPE: Starting recipe terms.to-patterns", locals())
    if dataset is None:
        log("RECIPE: Reading input terms from sys.stdin")
        terms = (json.loads(line) for line in sys.stdin)
    else:
        terms = DB.get_dataset(dataset)
        log("RECIPE: Reading {} input terms from dataset {}"
            .format(len(terms), dataset))
    if output_file:
        patterns = [get_pattern(term, label) for term in terms
                    if term['answer'] == 'accept']
        log("RECIPE: Generated {} patterns".format(len(patterns)))
        write_jsonl(output_file, patterns)
        prints("Exported {} patterns".format(len(patterns)), output_file)
    else:
        log("RECIPE: Outputting patterns")
        for term in terms:
            if term['answer'] == 'accept':
                print(json.dumps(get_pattern(term, label)))
