import random
import copy
from collections import defaultdict
from typing import Union, Optional
from pathlib import Path
import prodigy
from prodigy.components.db import connect
from prodigy.components.sorters import Probability
from prodigy.components.preprocess import add_tokens
from prodigy.util import set_hashes, get_labels
from prodigy.components.loaders import JSONL
import spacy
from spacy.tokens import Doc
from wasabi import msg

def get_seeds(seed_data):
    """
        A helper function to resolve the command line input of the `--seeds` argument
        to a dict of terms.

        seed_data (unicode): The seed terms, either a string, string path or None.
        RETURNS (dict): A dictionary of seed terms.
    """
    seed_terms = defaultdict(set)
    seed_path = Path(seed_data)
    if seed_path.is_file():
        seed_patterns = list(JSONL(seed_path))
        msg.text(f"Using {len(seed_patterns)} seed(s) from {seed_path}.")
        for pattern in seed_patterns:
            # Populate seed_terms dict from patterns, replacing white space for underscore in
            # MWEs to match the MWEs' convention in the vocabulary.
            seed_terms[pattern["label"].upper()].add(pattern["pattern"].strip().replace(" ","_"))
        return seed_terms
    seeds = [t.strip() for t in seed_data.split(",") if t != ""]
    seed_terms["no_label"] = seeds
    msg.text(f"Using {len(seeds)} seed(s): {', '.join(seeds)}")
    return seed_terms

def add_spans(stream):
    """
        Add spans to task for pre-higlighting terms with the best category
        given the current semantic model.
    """
    for eg in stream:
        task = copy.deepcopy(eg)
        if task["label"] != "no_label": # Do not pre-highlight if there is no label.
            spans=[{
                "token_start": task["tokens"][0]["id"],
                "token_end": task["tokens"][-1]["id"],
                "start": task["tokens"][0]["start"],
                "end": task["tokens"][-1]["end"],
                "text": task["text"],
                "label": task["label"],
                }]
            task["spans"] = spans
            # Rehash the newly created task so that hashes reflect added data.
            task = set_hashes(task)
        yield task

# Recipe decorator with argument annotations: (description, argument type,
# shortcut, type / converter function called on value before it's passed to
# the function). Descriptions are also shown when typing --help.
@prodigy.recipe(
    "terms.correct",
    dataset=("The dataset to use", "positional", None, str),
    vectors=("Loadable spaCy model with word vectors", "positional", None, str),
    seeds=("One or more comma-separated seed terms or patterns file", "option", "s", get_seeds),
    labels=("One or more comma-separated labels or txt file with one label per line", "option", "l", get_labels),
)
def terms_correct(dataset: str, vectors: str, seeds:  Optional[Union[str, Path]], labels:  Optional[Union[str, Path]]):
    """
        Bootstrap a terminology list with word vectors and seed terms. Prodigy
        will suggest similar terms based on the word vectors, and update the
        target semantic model accordingly. The suggestions from the semantic model
        can be edited manually by highlighting a substring of interest.
        This is especially useful when working with multiword expressions. If labels
        are provided, they will be suggested as well with the possibility to correct them
        (remove the label or select another label from the list).
    """

    # Load the spaCy model with vectors
    nlp = spacy.load(vectors)
    for s in nlp.vocab.vectors:
        nlp.vocab[s]
    # Connect to the database using the settings from prodigy.json and add the
    # seed terms to the dataset.
    DB = connect()
    if labels:
        if len(labels) == 1 and len(seeds) == 1:
            # Substitute the placeholder label for seeds collected from command line
            # if there is only one label provided.
            seeds[labels[0]] = seeds.pop("no_label")
    else:
        labels = list(seeds.keys())

    seed_tasks=[]
    for label, seed_set in seeds.items():
        for s in seed_set:
            if label != "no_label":
                # add spans if a meaningful label exists to keep the DB record
                # consistent with the entries added during the annotation procees.
                doc = nlp(s)
                span = doc[:]
                spans = [{"token_start": span.start,
                          "token_end": span.end,
                          "start": span.start_char,
                          "end": span.end_char,
                          "text": span.text,
                          "label": label}]
                seed_tasks.append(set_hashes({"text": s,
                                              "answer": "accept",
                                              "label": label,
                                              "spans": spans}))
            else:
                seed_tasks.append(set_hashes({"text": s,
                                              "answer": "accept",
                                              "label": label}))
    DB.add_examples(seed_tasks, datasets=[dataset])

    # Initialize containers for the accepted terms and the rejected terms.
    # Accepted terms are stored as spaCy doc objects keyed by label to easily
    # compute just one positive vector per class.
    # Rejected terms are stored as a set of spaCy lex objectes keyed by class, as we
    # will sample from rejected vector space when computing the similarity score in `predict` function.
    accept_dict={}
    for label, terms in seeds.items():
        if label in labels or label == "no_label":
            accept_dict[label] = Doc(nlp.vocab, words=terms)
    reject_dict = defaultdict(set)
    score = 0

    def compute_similarity(term, sample):
        """
            Computes pairwise similarity between a term vector and a sample of vectors
            using spaCy's .similarity() method.
        """
        results = []
        for lex in sample:
            if lex.vector_norm != 0.0:
                results.append((max(term.similarity(lex), 0.0), lex.text))
        return max(results)

    def get_negative_term_score(term, model_terms):
        """Computes the score for a term by sampling the negative vector set"""
        results = [] # list of (score, term, class) tuples
        max_score = 0.0
        for category, terms in model_terms.items():
            if len(terms) > 3:
                for _ in range(5):
                    random_sample = random.sample(list(terms), 3)
                    score, closest_term = compute_similarity(term, random_sample)
                    results.append((score, closest_term, category))
            else:
                random_sample = list(terms)
                score, closest_term = compute_similarity(term, random_sample)   
                results.append((score, closest_term, category))
        results.sort(key=lambda tup: tup[0], reverse=True)
        max_score = results[0][0]
        max_cat = results[0][2]
        return max_score, max_cat

    def get_positive_term_score(term, model_docs):
        """Computes the score for a term by comparing the term with the vector of current accepted terms."""
        results = []
        max_score = 0.0
        for category, category_doc in model_docs.items():
            if category_doc.vector_norm != 0.0:
                results.append((max(term.similarity(category_doc), 0.0), category))
        results.sort(key=lambda tup: tup[0], reverse=True)
        max_score = results[0][0]
        max_cat = results[0][1]
        return max_score, max_cat

    def predict(term):
        nonlocal accept_dict, reject_dict
        """Score a term by comparing it to the current semantic model."""
        reject_cat_suggestion = None
        accept_cat_suggestion = None
        if len(accept_dict) == 0 and len(reject_dict) == 0:
            return 0.5, None
        if len(accept_dict):
            accept_score, accept_cat_suggestion = get_positive_term_score(term, accept_dict)
        else:
            accept_score = 0.0
        if len(reject_dict):
            reject_score, reject_cat_suggestion = get_negative_term_score(term, reject_dict)
        else:
            reject_score = 0.0
        score = accept_score / (accept_score + reject_score + 0.2)
        category_suggestion = accept_cat_suggestion if accept_score >= reject_score else reject_cat_suggestion
        return max(score, 0.0), category_suggestion

    def update(answers):
        # Called whenever Prodigy receives new annotations.
        nonlocal accept_dict, reject_dict, score
        # Initialize containers for new terms.
        reject_words = set()
        accept_words = set()
        for answer in answers:
            # Increase or decrease score depending on answer and update
            # list of accepted and rejected terms.
            # In case of manual edition, collect new terms from spans.
            if answer["answer"] == "accept":
                score += 1
                spans = answer.get("spans", None)
                if spans:
                    for span in answer["spans"]:
                        tokens = answer["tokens"][span["token_start"]:span["token_end"]+1]
                        label = span["label"]
                        accept_words.add(("_".join([t["text"] for t in tokens]), label))
                else:
                    accept_words.add((answer["text"], answer["label"]))
            elif answer["answer"] == "reject":
                score -= 1
                reject_words.add((answer["text"],answer["label"]))
        
        # Update the target vector models in place
        for cat, cat_doc in accept_dict.items():
            cat_terms = set([t.text for t in cat_doc])
            cat_terms.update(term for term, label in accept_words if label == cat)
            accept_dict[cat] = Doc(nlp.vocab, words=cat_terms)
        for term,label in reject_words:
            reject_dict[label].add(nlp(term)[0].lex)

    def score_stream(stream):
        # Get all lexemes in the vocab and score them
        lexemes = [lex for lex in stream if lex.is_lower]
        while True:
            seen = set(w.orth for cat, cat_doc in accept_dict.items() for w in cat_doc)
            seen.update(set(w.orth for cat,terms in reject_dict.items() for w in terms))
            lexemes = [w for w in lexemes if w.orth not in seen and w.vector_norm]
            # if there are many seed terms a pairwise similarity operation on arrays should be faster
            by_score = [(predict(lex), lex) for lex in lexemes]
            by_score.sort(reverse=True)
            for _, term in by_score:
                score, cat = predict(term)
                # Return (score, example) tuples for the scored terms
                yield score, {"text": term.text.replace("_"," "),"label": cat,"meta": {"score": score}}

    # Sort the scored vocab by probability and return examples
    stream = Probability(score_stream(nlp.vocab))
    # Add tokens and spans to allow for highlighting sub strings
    stream = add_tokens(nlp, stream)
    stream = add_spans(stream)

    return {
        "view_id": "ner_manual",  # Annotation interface to use
        "dataset": dataset,  # Name of dataset to save annotations
        "stream": stream,  # Incoming stream of examples
        "update": update,  # Update callback, called with answers
        "config": {  # Additional config settings, mostly for app UI
            "lang": nlp.lang,
            "labels": labels,  # Selectable label options
        },
    }
