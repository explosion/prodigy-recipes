# coding: utf8
import prodigy
import spacy
from prodigy.components.loaders import JSONL
from prodigy.components.preprocess import add_tokens
from prodigy.util import split_string

"""
Custom recipe to annotate a dataset suitable to train an extractive Question Answering system.

As in SQuAD, each sample contains a question, a text (i.e., a paragraph in natural language, related to the question and likely to contain a possible answer) and the specific answer, extracted directly from the context. Notice that the answer is a substring of the context.

To make the interface show both the question and the text, you need to add the following custom Javascript code to your ~/.prodigy/prodigy.json file:

    {
    "javascript": "document.addEventListener('prodigyupdate', event => {const container = document.querySelector('.prodigy-title'); container.innerHTML=window.prodigy.content.question; });document.addEventListener('prodigymount', event => {const container = document.querySelector('.prodigy-title'); container.innerHTML=window.prodigy.content.question; })"
    }
"""


@prodigy.recipe(
    "qa",
    dataset=("The dataset to use", "positional", None, str),
    spacy_model=("The base model", "positional", None, str),
    source=("The source data as a JSONL file", "positional", None, str),
    label=("One or more comma-separated labels", "option", "l", split_string),
)
def qa(dataset, spacy_model, source, label="answer_span"):
    # load the source dataset, made of samples containing question and text pairs
    stream = JSONL(source)
    # load a spaCy model
    nlp = spacy.load(spacy_model)
    # and use it to tokenize the text
    stream = add_tokens(nlp, stream)

    return {
        "view_id": "ner_manual",
        "dataset": dataset,
        "stream": stream,
        "config": {"lang": nlp.lang, "label": label, "labels": label},
    }
