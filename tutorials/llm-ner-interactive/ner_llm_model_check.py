from pathlib import Path
from typing import Iterable, Optional, Union, List

import srsly
from spacy.cli._util import parse_config_overrides
from spacy.util import load_config
from spacy_llm.tasks import NERTask
from spacy_llm.util import assemble, assemble_from_config

from prodigy.components.db import connect
from prodigy.components.filters import filter_seen_before
from prodigy.components.preprocess import (
    add_tokens,
    get_llm_task,
    make_ner_suggestions,
    msg,
    resolve_labels,
    split_sentences,
)
from prodigy.components.stream import _source_is_dataset, get_stream
from prodigy.core import recipe, Controller
from prodigy.protocols import ControllerComponentsDict, RecipeEventHookProtocol
from prodigy.types import StreamType, TaskType
from prodigy.util import DEFAULT_LLM_BATCH_SIZE, get_timestamp_session_id, log


root = Path(__file__).parent


def add_model_check_suggestions(stream: StreamType) -> StreamType:
    for task in stream:
        task["field_id"] = "model"
        task["field_label"] = "Bigger model to check"
        task["field_suggestions"] = [
            "spacy.GPT-3-5.v1",
            "spacy.GPT-4.v1"
        ]
        yield task


@recipe(
    # fmt: off
    "ner.llm.correct-model-check",
    dataset=("Dataset to save answers to", "positional", None, str),
    config_path=("Path to the spacy-llm config file", "positional", None, str),
    source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    loader=("Loader (guessed from file extension if not set)", "option", "lo", str),
    segment=("Split articles into sentences", "flag", "S", bool),
    component=("Name of the component to use for annotation", "option", "c", str),
    # fmt: on
)
def llm_correct_ner_model_check(
    dataset: str,
    config_path: Path,
    source: Union[str, Iterable[dict]],
    loader: Optional[str] = None,
    segment: bool = False,
    component: str = "llm",
    _extra: List[str] = [],
) -> ControllerComponentsDict:
    """
    Perform zero- or few-shot annotation with the aid of large language models.
    """
    log("RECIPE: Starting recipe ner.llm.correct", locals())
    config_overrides = parse_config_overrides(list(_extra)) if _extra else {}
    config_overrides[f"components.{component}.save_io"] = True
    # In case of API auth errors the following call to `assemble` will throw a UserWarning
    # rather than an Exception, which makes it hard for us to gracefully handle here.
    nlp = assemble(config_path, overrides=config_overrides)
    llm_task = get_llm_task(nlp, component)
    if not isinstance(llm_task, NERTask):
        msg.fail(
            "Invalid spacy-llm Task Type for recipe 'ner.llm.correct'."
            f"Expected task type: {NERTask}. "
            f"Provided task type: {type(llm_task)}. ",
            "Modify your spacy-llm config or use a different LLM recipe.",
            exits=1,
        )
    labels = resolve_labels(nlp, component)

    stream = get_stream(
        source,
        loader=loader,
        rehash=True,
        dedup=True,
        input_key="text",
    )

    if segment:
        nlp.add_pipe("sentencizer")
        stream.apply(split_sentences, nlp=nlp, stream=stream)

    stream.apply(add_tokens, nlp=nlp, stream=stream)

    stream.apply(
        make_ner_suggestions,
        nlp=nlp,
        component=component,
        labels=labels,
        batch_size=DEFAULT_LLM_BATCH_SIZE,
    )

    stream.apply(add_model_check_suggestions)

    def check_big_llm(ctrl: Controller, *, task: TaskType):
        model_name = task.get("model", "spacy.GPT-4.v1")
        config = load_config(config_path, overrides=config_overrides)
        config["components"]["llm"]["model"] = {
            "@llm_models": model_name,
            "config": {
                "temperature": 0.0
            }
        }
        new_nlp = assemble_from_config(config)
        preds = make_ner_suggestions([task], new_nlp, component, labels, DEFAULT_LLM_BATCH_SIZE)
        new_task = next(preds)
        new_task["meta"]["model"] = model_name
        return new_task

    return {
        "dataset": dataset,
        "view_id": "blocks",
        "stream": stream,
        "config": {
            "batch_size": DEFAULT_LLM_BATCH_SIZE,
            "labels": labels,
            "exclude_by": "input",
            "blocks": [
                {"view_id": "ner_manual"},
                {"view_id": "llm_io"},
                {"view_id": "text_input"},
                {"view_id": "html", "html_template": (root / "ner_llm_model_check.html").read_text()},
            ],
            "javascript": (root / "ner_llm_model_check.js").read_text()
        },
        "event_hooks": {
            "check_big_model": check_big_llm
        }
    }
