from typing import List, Optional
import spacy
import prodigy
from prodigy.components.loaders import JSONL
from prodigy.components.preprocess import add_tokens
from prodigy.models.matcher import PatternMatcher
from prodigy.util import split_string


# Helper function for removing token information from examples
# before they're placed in the database. Used if character highlighting is enabled.
def remove_tokens(answers):
    for eg in answers:
        del eg["tokens"]
        if "spans" in eg:
            for span in eg["spans"]:
                del span["token_start"]
                del span["token_end"]
    return answers


# Recipe decorator with argument annotations: (description, argument type,
# shortcut, type / converter function called on value before it's passed to
# the function). Descriptions are also shown when typing --help.
@prodigy.recipe(
    "ner.manual",
    dataset=("The dataset to use", "positional", None, str),
    spacy_model=("The base model", "positional", None, str),
    source=("The source data as a JSONL file", "positional", None, str),
    label=("One or more comma-separated labels", "option", "l", split_string),
    patterns=("The match patterns file","option","p",str),
    exclude=("Names of datasets to exclude", "option", "e", split_string),
    highlight_chars=("Allow for highlighting individual characters instead of tokens", "flag", "C", bool),
)
def ner_manual(
    dataset: str,
    spacy_model: str,
    source: str,
    label: Optional[List[str]] = None,
    patterns: Optional[str] = None,
    exclude: Optional[List[str]] = None,
    highlight_chars: bool = False,
):
    """
    Mark spans manually by token. Requires only a tokenizer and no entity
    recognizer, and doesn't do any active learning. If patterns are provided,
    their matches are highlighted in the example, if available. The patterns file can
    include exact strings or token patterns for use with spaCy's `Matcher`.
    The recipe will present all examples in order, so even examples without matches are shown.
    If character highlighting is enabled, no "tokens" are saved to the database.
    """
    # Load the spaCy model for tokenization.
    nlp = spacy.load(spacy_model)

    # Load the stream from a JSONL file and return a generator that yields a
    # dictionary for each example in the data.
    stream = JSONL(source)

    # If patterns are provided, apply matcher to the stream, which returns (score, example) tuples.
    # `all_examples=True` will display all examples, including the ones without any matches and
    # `combine_matches=True` will show all matches in one task as opposed to splitting them to different tasks.
    if patterns is not None:
        pattern_matcher = PatternMatcher(nlp, combine_matches=True, all_examples=True)
        pattern_matcher = pattern_matcher.from_disk(patterns)
        stream = (eg for _,eg in pattern_matcher(stream))

    # Tokenize the incoming examples and add a "tokens" property to each
    # example. Also handles pre-defined selected spans. Tokenization allows
    # faster highlighting, because the selection can "snap" to token boundaries.
    # If `use_chars` is True, tokens are split into individual characters, which enables
    # character based selection as opposed to default token based selection.
    stream = add_tokens(nlp, stream, use_chars=highlight_chars)

    return {
        "view_id": "ner_manual",  # Annotation interface to use
        "dataset": dataset,  # Name of dataset to save annotations
        "stream": stream,  # Incoming stream of examples
        "exclude": exclude,  # List of dataset names to exclude
        "before_db": remove_tokens if highlight_chars else None,
        # Remove token information to permit highlighting individual characters
        "config": {  # Additional config settings, mostly for app UI
            "lang": nlp.lang,
            "labels": label,  # Selectable label options
        },
    }
