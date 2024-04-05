import copy
from typing import Dict, Iterable, List, Callable, Tuple, Optional, Any
from prodigy.components.stream import get_stream
from prodigy.core import recipe, Arg
from prodigy.types import StreamType
from prodigy.protocols import ControllerComponentsDict
from prodigy.util import log
from prodigy.components.preprocess import split_sentences
import spacy
from prodigy.core import Controller
from collections import Counter
from wasabi import msg
from prodigy.util import set_hashes
from prodigy.recipes.train import setup_gpu
from prodigy.components.preprocess import (
    make_ner_suggestions,
    make_spancat_suggestions,
    resolve_labels,
)


def get_errors(gold: Dict, pred: Dict) -> Dict:
    """
    Find false positives, false negatives and true positives for a given example.
    Args:
        gold: Dict, gold example
        pred: Dict, predicted example
    Returns:
        errors: Dict, dictionary of erroneouns spans with keys "FP", "FN", "TP"
    """
    errors: Dict[str, List[Dict[Any, Any]]] = {"FP": [], "FN": [], "TP": []}
    all_spans = set([tuple(d.items()) for d in gold["spans"]]).union(
        set([tuple(d.items()) for d in pred["spans"]])
    )

    for span_tuple in all_spans:
        span = dict(span_tuple)
        in_gold = span in gold["spans"]
        in_pred = span in pred["spans"]

        if in_gold and in_pred:
            errors["TP"].append(span)
        elif in_gold:
            errors["FN"].append(span)
        elif in_pred:
            errors["FP"].append(span)

    return errors


def annotate_errors(gold: Dict, pred: Dict) -> List[Dict]:
    """
    Generate a list of tasks for Prodigy to review the
    false positives and false negatives.
    There is one Prodigy task per error.
    Args:
        gold: Dict, gold example
        pred: Dict, predicted example
    Returns:
        tasks: List[Dict], list of tasks with one error per task
    """
    tasks = []

    # Ensure text match between gold and pred examples
    assert gold["text"] == pred["text"], "Mismatch in gold and pred example."

    # Get errors
    errors = get_errors(gold, pred)

    for error_type, error_spans in errors.items():
        for error in error_spans:
            spans = []
            if error_type == "FN":
                context = pred
                label_suffix = "_PRED"
                error_context = gold
                error_suffix = "_GOLD_FN"
                context_label_suffix = "_GOLD"
            elif error_type == "FP":
                # look for error in pred spans
                context = gold
                label_suffix = "_GOLD"
                error_context = pred
                error_suffix = "_PRED_FP"
                context_label_suffix = "_PRED"
            elif error_type == "TP":
                # ignore true positives
                continue
            else:
                msg.fail(f"Unknown error type: {error_type}", exits=1)
            # Iterate over spans in the context
            for span in error_context["spans"]:
                s = copy.deepcopy(span)
                if s == error:
                    s["label"] = f"{span['label']}{error_suffix}"
                else:
                    s["label"] = f"{span['label']}{context_label_suffix}"
                spans.append(s)
            for span in context["spans"]:
                s = copy.deepcopy(span)
                s["label"] = f"{span['label']}{label_suffix}"
                spans.append(s)

            # Create new task
            new_task = copy.deepcopy(gold)
            new_task["spans"] = spans
            new_task["error"] = error_type
            task = set_hashes(new_task, task_keys=["spans", "error"], overwrite=True)
            tasks.append(task)

    return tasks


def get_printer() -> Callable[[Controller], None]:
    """
    Create a function to print the results of the error analysis.
    Returns:
        _format_printer: Callable, function to print the results.
    """

    def _format_printer(ctrl: Controller) -> None:
        examples = ctrl.db.get_dataset_examples(ctrl.dataset)
        fp_counts: Counter = Counter()
        fn_counts: Counter = Counter()
        fp_answers = []
        fn_answers = []
        # Get counts per error type
        for eg in examples:
            if eg["answer"] == "accept":
                answers = eg["accept"]
                reason = eg.get("reason")
                if eg["error"] == "FP":
                    fp_answers.extend(answers)
                    if reason:
                        fp_answers.append(reason)
                elif eg["error"] == "FN":
                    fn_answers.extend(answers)
                    if reason:
                        fn_answers.append(reason)
                else:
                    msg.warn(f"Unknown error type: {eg['error']}")
        fp_counts.update(fp_answers)
        fn_counts.update(fn_answers)
        print("")  # noqa: T201
        log("RECIPE: Calculating results")
        msg.divider("Evaluation results for false positives", icon="emoji")
        rows = []
        for reason, count in fp_counts.items():
            row = (reason, count)
            rows.append(row)
        msg.table(rows, header=["Reason", "Count"], aligns=("l", "r"))
        msg.divider("Evaluation results for false negatives", icon="emoji")
        rows = []
        for reason, count in fn_counts.items():
            row = (reason, count)
            rows.append(row)
        msg.table(rows, header=["Reason", "Count"], aligns=("l", "r"))

    return _format_printer


def set_default_label_colors(labels: List[str]) -> Dict[str, str]:
    """
    Set default colors for labels in Prodigy.
    There is a diferent color for predicted and gold spans.
    The errors are colored in red.
    Args:
        labels: List[str], list of project labels
    Returns:
        default_theme: Dict[str, str], dictionary with the default colors for the labels
    """
    default_theme = {}
    for label in labels:
        default_theme[label + "_PRED"] = "#DAF7A6"
        default_theme[label + "_GOLD"] = "#FFC300"
        default_theme[label + "_PRED_FP"] = "#FF5733"
        default_theme[label + "_GOLD_FN"] = "#FF5733"
    return default_theme


def add_options(stream: StreamType) -> Iterable[Dict]:
    """
    Add options and freeform textfield to the task to specify the reason for the error.
    Args:
        stream: StreamType, Prodigy stream
    Yields:
        new_task: Dict, task with options to select the reason for the error.
    """
    for task in stream:
        new_task = copy.deepcopy(task)
        options = [
            {"id": "gold_error", "text": "gold annotation error"},
            {"id": "preprocessing", "text": "boundary error due to preprocessing"},
            {"id": "tokenization", "text": "boundary error due to tokenization"},
            {"id": "boundary", "text": "other boundary error"},
            {"id": "classification", "text": "other classification error"},
        ]
        new_task["options"] = options
        new_task["field_id"] = "reason"
        new_task["field_placeholder"] = "Other"
        yield new_task


def filter_labels(stream: StreamType, labels: List[str]) -> Iterable[Dict]:
    """
    Filter the stream to keep only the spans with the labels
    specified in the recipe CLI.
    Args:
        stream: StreamType, Prodigy stream
        labels: List[str], list of labels to keep
    Yields:
        new_task: Dict, task with only the labels specified in the recipe CLI.
    """
    for task in stream:
        new_task = copy.deepcopy(task)
        new_task["spans"] = [span for span in task["spans"] if span["label"] in labels]
        yield new_task


@recipe(
    # fmt: off
    "error.analysis",
    dataset=Arg(help="Dataset to save annotations to"),
    model=Arg(help="spaCy model to evaluate"),
    ner=Arg("--ner", "-n", help="NER evaluation dataset"),
    spancat=Arg("--spancat", "-s", help="Spancat evaluation datasets"),
    labels=Arg("--labels", "-l", help="Comma separated list of labels to analyze"),
    segment=Arg("--split","-S", help="Split articles into sentences"),
    gpu_id=Arg("--gpu", "-GPU", help="GPU ID. Defaults to -1 i.e. CPU"),
    # fmt: on
)
def error_analysis(
    dataset: str,
    model: str,
    ner: Optional[str] = None,
    spancat: Optional[str] = None,
    labels: Optional[List[str]] = None,
    segment: bool = False,
    gpu_id: int = -1,
) -> ControllerComponentsDict:
    """
    Iterate through false postives and false negatives
    and collect counts of reasons for errors to inform model improvement.
    """
    log("RECIPE: Starting recipe error_analysis", locals())
    setup_gpu(gpu_id)
    nlp = spacy.load(model)
    eval_dataset: Optional[str] = None
    component: Optional[str] = None
    if ner:
        component = "ner"
        eval_dataset = ner
    elif spancat:
        component = "spancat"
        eval_dataset = spancat
    if ner and spancat:
        msg.fail("Please specify either NER or spancat for error analysis.")
    if eval_dataset is None:
        msg.fail(
            "No evaluation dataset was specified. Please specify --ner or --spancat evaluation dataset."
        )
    labels = resolve_labels(nlp, component, labels)
    assert labels is not None
    DEFAULT_LABEL_COLORS = set_default_label_colors(labels)

    gold_stream = get_stream(
        f"dataset:{eval_dataset}", rehash=False, dedup=False, input_key="text"
    )
    gold_stream.apply(filter_labels, labels=labels)
    gold_examples = list(gold_stream)

    inputs = (eg for eg in gold_examples)
    if ner:
        pred_stream = make_ner_suggestions(
            stream=inputs, nlp=nlp, component=component, labels=labels
        )
    elif spancat:
        pred_stream = make_spancat_suggestions(
            stream=inputs, nlp=nlp, component=component, labels=labels
        )
    else:
        msg.fail("Please specify either NER or spancat for error analysis.")

    annotation_pairs: List[Tuple[Dict, Dict]] = list(
        zip(gold_examples, list(pred_stream))
    )
    stream_lst = []
    for pair in annotation_pairs:
        gold_example, pred_example = pair
        # we'll be comparing span dictionaries so we need to remove the source
        for span in gold_example["spans"]:
            span.pop("source", None)
        for span in pred_example["spans"]:
            span.pop("source", None)
        # generate task list, one task per error
        tasks = annotate_errors(gold_example, pred_example)
        stream_lst.extend(tasks)
    # convert to Prodigy StreamType
    stream = get_stream(stream_lst)
    stream.apply(add_options)
    if segment:
        nlp.add_pipe("sentencizer")
        stream.apply(split_sentences, nlp=nlp, stream=stream)

    custom_theme = {"labels": DEFAULT_LABEL_COLORS}
    on_exit = get_printer()

    return {
        "dataset": dataset,
        "view_id": "blocks",
        "stream": stream,
        "on_exit": on_exit,
        "config": {
            "exclude_by": "task",
            "blocks": [
                {"view_id": "spans"},
                {"view_id": "choice", "text": None},
                {"view_id": "text_input"},
            ],
            "choice_style": "multiple",
            "custom_theme": custom_theme,
        },
    }
