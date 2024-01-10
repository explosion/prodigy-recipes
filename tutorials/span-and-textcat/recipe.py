import spacy
import prodigy 
from prodigy.components.preprocess import add_tokens
from prodigy.components.loaders import JSONL


@prodigy.recipe(
    "span-and-textcat",
    dataset=("Dataset to save annotations into", "positional", None, str),
    lang=("Language to use", "positional", None, str),
    file_in=("Path to examples.jsonl file", "positional", None, str)
)
def custom_recipe(dataset, lang, file_in):
    span_labels = ["product", "amount", "size", "type", "topping"]
    textcat_labels = ["greet", "inform", "purchase", "confirm"]

    def add_options(stream):
        for ex in stream:
            ex['options'] = [
                {"id": lab, "text": lab} for lab in textcat_labels
            ]
            yield ex

    nlp = spacy.blank(lang)
    stream = JSONL(file_in)
    stream = add_tokens(nlp, stream)

    stream = add_options(stream)
    blocks = [
        {"view_id": "spans_manual"},
        {"view_id": "choice", "text": None},
    ]
    return {
        "view_id": "blocks",  # Annotation interface to use
        "dataset": dataset,  # Name of dataset to save annotations
        "stream": stream,  # Incoming stream of examples
        "config": {  # Additional config settings, mostly for app UI
            "lang": nlp.lang,
            "labels": span_labels,
            "blocks": blocks,
            "keymap_by_label": {
                "0": "q", 
                "1": "w", 
                "2": "e", 
                "3": "r", 
                "product": "1", 
                "amount": "2",
                "size": "3",
                "type": "4",
                "topping": "5" 
            },
            "choice_style": "multiple"
        },
    }
