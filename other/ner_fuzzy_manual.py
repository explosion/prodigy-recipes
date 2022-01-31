from typing import List, Optional
import copy
from collections import defaultdict
import prodigy
from prodigy.components.loaders import JSONL
from prodigy.components.preprocess import add_tokens
from prodigy.util import split_string, set_hashes
from prodigy.models.matcher import parse_patterns
import spacy
from spacy.tokens import Span
from spaczz.matcher import FuzzyMatcher


def apply_fuzzy_matcher(stream, nlp, fuzzy_matcher, pattern_labels, line_numbers):
    """Add a 'spans' key to each example, with fuzzy pattern matches."""
    # Process the stream using spaCy's nlp.pipe, which yields doc objects.
    # If as_tuples=True is set, you can pass in (text, context) tuples.
    texts_examples = ((eg["text"], eg) for eg in stream)
    for doc, eg in nlp.pipe(texts_examples, as_tuples=True):
        task = copy.deepcopy(eg)
        matched_spans = []
        for match_id, start_token, end_token, _ in fuzzy_matcher(doc):
            span_obj = Span(doc, start_token, end_token)
            span = {
                        "text": span_obj.text,
                        "start": span_obj.start_char,
                        "end": span_obj.end_char,
                        "pattern": span_obj.label,
                        "token_start": span_obj.start,
                        "token_end": span_obj.end - 1,
                        "label": pattern_labels[match_id],
                        "pattern_hash": match_id
                    }
            matched_spans.append(span)
        if matched_spans:
            task["spans"] = matched_spans
            all_ids = [line_numbers[s["pattern_hash"]]+1 for s in task["spans"]]
            task["meta"]["pattern"] = ", ".join([f"{p}" for p in all_ids])
            task = set_hashes(task)
        yield task


# Recipe decorator with argument annotations: (description, argument type,
# shortcut, type / converter function called on value before it's passed to
# the function). Descriptions are also shown when typing --help.
@prodigy.recipe(
    "ner.fuzzy.manual",
    dataset=("The dataset to use", "positional", None, str),
    spacy_model=("The base model", "positional", None, str),
    source=("The source data as a JSONL file", "positional", None, str),
    patterns=("Phrase patterns", "positional", None, str),
    label=("One or more comma-separated labels", "option", "l", split_string),
    exclude=("Names of datasets to exclude", "option", "e", split_string),
    
)
def ner_fuzzy_manual(
    dataset: str,
    spacy_model: str,
    source: str,
    patterns: str,
    label: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    
):
    """
    Mark spans manually by token with suggestions from spaCy phrase patterns pre-highlighted.
    The suggestions are entity spans matched by spaczz fuzzy matcher ignoring the case.
    Note, that if spaCy token patterns are required, spaczz syntax for token patterns should be observed
    and a custom parsing function should be implemented. Please check spaczz documentation
    for details: https://spacy.io/universe/project/spaczz.
    The recipe doesn't require any entity recognizer, and it doesn't do any active learning.
    It will present all examples in order, so even examples without matches are shown.
    """
    # Load the spaCy model for tokenization
    nlp = spacy.load(spacy_model)

    # Initialize spaczz fuzzy matcher
    fuzzy_matcher = FuzzyMatcher(nlp.vocab)

    # Load phrase patterns and feed them to spaczz matcher
    patterns = JSONL(patterns)
    _, phrase_patterns, line_numbers = parse_patterns(list(patterns))
    pattern_labels = defaultdict(str)
    for pattern_label, patterns in phrase_patterns.items():
        for (pattern_hash, pattern) in patterns:
            fuzzy_matcher.add(pattern_hash, [nlp(pattern)], kwargs= [{"ignorecase": True}])
            # Build pattern_hash to pattern map to recover the source pattern for UI.
            pattern_labels[pattern_hash] = pattern_label

    # Load the stream from a JSONL file and return a generator that yields a
    # dictionary for each example in the data.
    stream = JSONL(source)

    # Tokenize the incoming examples and add a "tokens" property to each
    # example. Also handles pre-defined selected spans. Tokenization allows
    # faster highlighting, because the selection can "snap" to token boundaries.
    stream = add_tokens(nlp, stream)

    # Apply the spaczz matcher to the stream and add matched spans to each task for pre-highlighting.
    stream = apply_fuzzy_matcher(stream, nlp, fuzzy_matcher, pattern_labels, line_numbers)

    return {
        "view_id": "ner_manual",  # Annotation interface to use
        "dataset": dataset,  # Name of dataset to save annotations
        "stream": stream,  # Incoming stream of examples
        "exclude": exclude,  # List of dataset names to exclude
        "config": {  # Additional config settings, mostly for app UI
            "lang": nlp.lang,
            "labels": label
        },
    }
