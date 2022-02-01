from typing import List, Optional
import copy
from collections import defaultdict
import prodigy
from prodigy.components.loaders import JSONL
from prodigy.components.preprocess import add_tokens
from prodigy.util import split_string, set_hashes
import spacy
from spacy.tokens import Span
from spaczz.matcher import FuzzyMatcher


def parse_phrase_patterns(patterns):
    """
        A parser for patterns file.
        It assumes Prodigy's patterns format with string patterns such as {"pattern": "some pattern", "label": "SOME_LABEL"}.
    """
    phrase_patterns = defaultdict(list)
    # Auxiliary map to recover the pattern id as line number in the UI
    line_numbers = {}
    for i, entry in enumerate(patterns):
        label = entry["label"]
        pattern = entry["pattern"]
        line_number = i+1
        phrase_patterns[label].append((line_number, pattern))
        line_numbers[line_number] = label
    return phrase_patterns, line_numbers


def apply_fuzzy_matcher(stream, nlp, fuzzy_matcher, line_numbers):
    """Add a 'spans' key to each example, with fuzzy pattern matches."""
    # Process the stream using spaCy's nlp.pipe, which yields doc objects.
    # If as_tuples=True is set, you can pass in (text, context) tuples.
    texts_examples = ((eg["text"], eg) for eg in stream)
    for doc, eg in nlp.pipe(texts_examples, as_tuples=True):
        task = copy.deepcopy(eg)
        matched_spans = []
        for line_number, start_token, end_token, _ in fuzzy_matcher(doc):
            span_obj = Span(doc, start_token, end_token)
            span = {
                        "text": span_obj.text,
                        "start": span_obj.start_char,
                        "end": span_obj.end_char,
                        "token_start": span_obj.start,
                        "token_end": span_obj.end -1,
                        "label": line_numbers[line_number],
                        "line_number": line_number
                    }
            matched_spans.append(span)
        if matched_spans:
            task["spans"] = matched_spans
            all_ids = []
            for s in task["spans"]:
                all_ids.append(s["line_number"])
                # Not needed anymore
                del s["line_number"]
            task["meta"]["pattern"] = ", ".join([f"{p}" for p in all_ids])
            # Rehash the newly created task so that hashes reflect added data
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
    Mark spans manually by token with suggestions from phrase patterns pre-highlighted.
    The suggestions are spans matched by spaczz fuzzy matcher ignoring the case.
    Note, that if token patterns are required, spaczz syntax for token patterns should be observed
    and a the parsing function should be modified accordingly. Please check spaczz documentation
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
    phrase_patterns, line_numbers = parse_phrase_patterns(list(patterns))
    for pattern_label, patterns in phrase_patterns.items():
        for (line_number, pattern) in patterns:
            # Use the line number from the patterns source file as the pattern_id
            fuzzy_matcher.add(line_number, [nlp(pattern)], kwargs= [{"ignorecase": True}])

    # Load the stream from a JSONL file and return a generator that yields a
    # dictionary for each example in the data.
    stream = JSONL(source)

    # Tokenize the incoming examples and add a "tokens" property to each
    # example. Also handles pre-defined selected spans. Tokenization allows
    # faster highlighting, because the selection can "snap" to token boundaries.
    stream = add_tokens(nlp, stream)

    # Apply the spaczz matcher to the stream.
    stream = apply_fuzzy_matcher(stream, nlp, fuzzy_matcher, line_numbers)

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
