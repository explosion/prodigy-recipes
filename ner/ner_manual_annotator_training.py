from re import L
from typing import Any, Dict, Iterable, Optional, List, Union
from functools import partial

import spacy

from prodigy.components.loaders import get_stream
from prodigy.models.matcher import PatternMatcher
from prodigy.core import recipe
from prodigy.types import RecipeSettingsType, TaskType
from prodigy.util import split_string, get_labels, msg, log, INPUT_HASH_ATTR
from prodigy.components.preprocess import add_tokens
from prodigy.recipes.ner import remove_tokens


def spans_equal(s1: Dict[str, Any], s2: Dict[str, Any]) -> bool:
    return s1["start"] == s2["start"] and s1["end"] == s2["end"]


def labels_equal(s1: Dict[str, Any], s2: Dict[str, Any]) -> bool:
    return s1["label"] == s2["label"]


def ensure_span_text(eg: TaskType) -> TaskType:
    for span in eg["spans"]:
        if "text" not in span:
            span["text"] = eg["text"][span["start"] : span["end"]]
    return eg


def validate_answer(answer: TaskType, *, known_answers: List[TaskType]):
    for known_answer in known_answers:
        known_answer = ensure_span_text(known_answer)
        if known_answer[INPUT_HASH_ATTR] == answer[INPUT_HASH_ATTR]:
            errors = []
            known_spans = known_answer.get("spans",[])
            answer_spans = answer.get("spans",[])
            explanation_label = known_answer.get("meta",{}).get("explanation_label")
            explanation_boundaries = known_answer.get("meta", {}).get("explanation_boundary")
            if not explanation_boundaries:
                explanation_boundaries = 'Se debe anotar el nombre completo sin signos de puntuación,  determinantes o modificadores, a no ser que forman parte del nombre propio. Para más información consulte Paso 2 en el documento \"Descripción de la tarea\"'

            if len(known_spans) > len(answer_spans):
                errors.append('Anotó menos entidades de las esperadas para esta respuesta. Todas las menciones deben estar anotadas. Para más información consulte el apartado C2 de la Guía'
                )
            elif len(known_spans) < len(answer_spans):
                errors.append(
                    'Anotó más entidades de las esperadas para esta respuesta.' 
                )
            for known_span, span in zip(known_answer.get("spans",[]), answer.get("spans",[])):
                if not spans_equal(known_span, span):
                    # boundary error
                    errors.append(explanation_boundaries)
                elif not labels_equal(known_span, span):
                    # label error
                    errors.append(explanation_label)

            if len(errors) > 0:
                error_msg = "\n".join(errors)
                if known_spans:
                    expected_spans = [
                        f'[{s["text"]}]: {s["label"]}' for s in known_spans
                    ]
                    error_msg += f"\n\nAnotaciones esperadas:"
                    if expected_spans:
                        error_msg += "\n"
                        for span_msg in expected_spans:
                            error_msg += span_msg + "\n"
                else:
                    error_msg += f"\n\nNo hay entidades para anotar"
                raise ValueError(error_msg)


@recipe(
    "ner.annotator-training",
    # fmt: off
    dataset=("Dataset to save annotations to", "positional", None, str),
    spacy_model=("Loadable spaCy pipeline for tokenization or blank:lang (e.g. blank:en)", "positional", None, str),
    source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    known_answers=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    loader=("Loader (guessed from file extension if not set)", "option", "lo", str),
    label=("Comma-separated label(s) to annotate or text file with one label per line", "option", "l", get_labels),
    patterns=("Path to match patterns file", "option", "pt", str),
    exclude=("Comma-separated list of dataset IDs whose annotations to exclude", "option", "e", split_string),
    highlight_chars=("Allow highlighting individual characters instead of tokens", "flag", "C", bool),
    # fmt: on
)
def annotator_training(
    dataset: str,
    spacy_model: str,
    source: Union[str, Iterable[dict]],
    known_answers: Union[str, Iterable[dict]],
    loader: Optional[str] = None,
    label: Optional[List[str]] = None,
    patterns: Optional[str] = None,
    exclude: Optional[List[str]] = None,
    highlight_chars: bool = False,
) -> RecipeSettingsType:
    """
    This recipe is the same as the standard ner.manual recipe but adds a `validate_answer`
    callback that ensures only 1 set of annotations will be expected based on the `known_answers`
    input. It can be used to help train annotators by giving them free reign to annotate an example
    but showing them why it's wrong if they don't set the correct annotations.
    """
    log("RECIPE: Starting recipe ner.annotator-training", locals())
    nlp = spacy.load(spacy_model)
    labels = label  # comma-separated list or path to text file
    if not labels:
        labels = nlp.pipe_labels.get("ner", [])
        if not labels:
            msg.fail("No --label argument set and no labels found in model", exits=1)
        msg.text(f"Using {len(labels)} labels from model: {', '.join(labels)}")
    log(f"RECIPE: Annotating with {len(labels)} labels", labels)
    stream = get_stream(
        source,
        loader=loader,
        rehash=True,
        dedup=True,
        input_key="text",
        is_binary=False,
    )
    known_answers = list(
        get_stream(
            known_answers,
            loader=loader,
            input_key="text",
            is_binary=False,
        )
    )
    if patterns is not None:
        pattern_matcher = PatternMatcher(nlp, combine_matches=True, all_examples=True)
        pattern_matcher = pattern_matcher.from_disk(patterns)
        stream = (eg for _, eg in pattern_matcher(stream))

    default_explanation_boundaries = 'Se debe anotar el nombre completo sin signos de puntuación,  determinantes o modificadores, a no ser que forman parte del nombre propio. Para más información consulte Paso 2 en el documento \"Descripción de la tarea\"'

    # Add "tokens" key to the tasks, either with words or characters
    stream = add_tokens(nlp, stream, use_chars=highlight_chars)

    return {
        "view_id": "ner_manual",
        "dataset": dataset,
        "stream": stream,
        "exclude": exclude,
        "validate_answer": partial(validate_answer, known_answers=known_answers),
        "before_db": remove_tokens if highlight_chars else None,
        "config": {
            "lang": nlp.lang,
            "labels": labels,
            "exclude_by": "input",
            "ner_manual_highlight_chars": highlight_chars,
            "auto_count_stream": True,
        },
    }
