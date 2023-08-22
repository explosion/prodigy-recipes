import copy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union
from spacy_llm.util import assemble

from prodigy.components.preprocess import (
    add_tokens,
    split_sentences,
)
from prodigy.components.stream import get_stream
from prodigy.core import recipe
from prodigy.protocols import ControllerComponentsDict
from prodigy.util import log, split_string
from prodigy.components.openai import GLOBAL_STYLE
from prodigy.types import StreamType
from spacy.language import Language

DEFAULT_BATCH_SIZE = 3

NER_HTML_TEMPLATE = """
    <div class="cleaned">
    {{#has_ent_errors}}
    <summary>Warning ⚠: The following entities were not found in the KB:</summary>
    <pre>{{ent_errors_html}}</pre>
    {{/has_ent_errors}}
    {{#has_label_errors}}
    <summary>
    Warning ⚠: The following entities were found in the KB but with different labels:
    </summary>
    <pre>{{label_errors_html}}</pre>
    {{/has_label_errors}}
    <details>
        <summary>Show the prompt sent to LLM</summary>
        <pre>{{llm.prompt}}</pre>
    </details>  
    <details>
        <summary>Show the response from the LLM</summary>
        <pre>{{llm.response}}</pre>
    </details>
    </div>
    """


def validate_predictions(stream: StreamType) -> StreamType:
    """Validate the llm predictions against the KB and add error information to task.

    stream (iterable): The stream of annotation tasks.
    YIELDS (dict): The annotation examples.
    """
    for eg in stream:
        # bool error flags facilitate conditional UI display & filtering
        eg["has_ent_errors"] = False
        eg["ent_errors"] = []
        eg["has_label_errors"] = False
        eg["label_errors"] = {}
        for llm_span in eg["spans"]:
            overlapping = False
            for kb_span in eg["kb_spans"]:
                if (
                    llm_span["start"] == kb_span["start"]
                    and llm_span["end"] == kb_span["end"]
                ):
                    overlapping = True
                    break

            if overlapping:
                # not all KB spans have types
                if kb_span["types"]:
                    # LLM labels should correspond to KB types
                    if llm_span["label"] not in kb_span["types"]:
                        eg["label_errors"][llm_span["text"]] = kb_span["types"]
                        eg["has_label_errors"] = True
            else:
                eg["ent_errors"].append(llm_span["text"])
                eg["has_ent_errors"] = True
        eg["label_errors_html"] = "\n".join(
            [f"{k}: {', '.join(v)}" for k, v in eg["label_errors"].items()]
        )
        eg["ent_errors_html"] = "\n".join(eg["ent_errors"])
        yield eg


def filter_validated(stream: StreamType) -> StreamType:
    """Filter out examples that are validated by KB.
    stream (iterable): The stream of annotation tasks.
    YIELDS (dict): The annotation examples.
    """
    for eg in stream:
        if eg["has_ent_errors"] or eg["has_label_errors"]:
            yield eg


def add_ner_annotations(
    stream: StreamType,
    nlp: Language,
    labels: Iterable[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> StreamType:
    """Add a 'spans' key to each example, with entities predicted by LLM
    and 'kb_spans' key, with entities matched by KB.

    stream (iterable): The stream of annotation tasks.
    nlp (Language): The spaCy pipeline.
    labels (Iterable[str]): The entity labels to annotate.
    batch_size (int): The batch size for the LLM.
    YIELDS (dict): The annotation examples.
    """
    texts = ((eg["text"], eg) for eg in stream)
    for doc, eg in nlp.pipe(texts, as_tuples=True, batch_size=batch_size):
        task = copy.deepcopy(eg)
        llm_spans = []
        kb_spans = []
        for ent in doc.ents:
            if labels and ent.label_ not in labels:
                continue
            llm_spans.append(
                {
                    "token_start": ent.start,
                    "token_end": ent.end - 1,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "text": ent.text,
                    "label": ent.label_,
                }
            )
        task["spans"] = llm_spans

        def parse_types(raw_results: Dict) -> List[str]:
            return [
                db_type.split(":")[1]
                for db_type in raw_results["@types"].split(",")
                if db_type.startswith("DBpedia")
            ]

        for ent in doc.spans["dbpedia_spotlight"]:
            kb_spans.append(
                {
                    "token_start": ent.start,
                    "token_end": ent.end - 1,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "text": ent.text,
                    "types": parse_types(ent._.dbpedia_raw_result),
                }
            )
        task["kb_spans"] = kb_spans
        task["llm"] = doc.user_data["llm_io"]["llm"]
        yield task


@recipe(
    # fmt: off
    "ner.llm.kb.correct",
    dataset=("Dataset to save answers to", "positional", None, str),
    config_path=("Path to the spacy-llm config file", "positional", None, str),
    source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    labels=("Comma-separated label(s) to annotate or text file with one label per line", "positional", None, split_string),
    loader=("Loader (guessed from file extension if not set)", "option", "lo", str),
    segment=("Split articles into sentences", "flag", "S", bool),
    skip_validated=("Skip examples that are validated by KB", "flag", "V", bool),
    # fmt: on
)
def llm_kb_correct_ner(
    dataset: str,
    config_path: Path,
    source: Union[str, Iterable[dict]],
    labels: List[str],
    loader: Optional[str] = None,
    segment: bool = False,
    skip_validated: bool = False,
) -> ControllerComponentsDict:
    """
    Perform zero- or few-shot annotation with the aid of large language models
    and validate with KB.
    """
    log("RECIPE: Starting recipe ner.llm.kb.correct", locals())
    config_overrides = {}
    config_overrides["components.llm.save_io"] = True
    config_overrides["components.dbpedia-spotlight.overwrite_ents"] = False
    # In case of API auth errors the following call to `assemble` will throw a UserWarning
    # rather than an Exception, which makes it hard for us to gracefully handle here.
    nlp = assemble(config_path, overrides=config_overrides)
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

    stream.apply(add_ner_annotations, nlp=nlp, labels=labels)
    stream.apply(validate_predictions)
    if skip_validated:
        stream.apply(filter_validated, stream=stream)
    return {
        "dataset": dataset,
        "view_id": "blocks",
        "stream": stream,
        "config": {
            "batch_size": DEFAULT_BATCH_SIZE,
            "labels": labels,
            "exclude_by": "input",
            "blocks": [
                {"view_id": "ner_manual"},
                {"view_id": "html", "html_template": NER_HTML_TEMPLATE},
            ],
            "global_css": GLOBAL_STYLE,
        },
    }
