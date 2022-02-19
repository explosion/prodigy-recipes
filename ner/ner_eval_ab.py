from collections import Counter
from typing import List, Optional
import random
from tabulate import tabulate
import spacy
import prodigy
from prodigy.components.loaders import JSONL
from prodigy.util import split_string
from prodigy.components.preprocess import split_sentences, set_hashes


def make_tasks(nlp, labels, stream):
    """
        Generate a task for each example in a stream so that it contains:
        a unique id, text input and model's predictions as output.
    """
    texts = ((eg["text"], eg) for eg in stream)
    # Process the stream using spaCy's nlp.pipe, which yields doc objects.
    # If as_tuples=True is set, you can pass in (text, context) tuples.
    for i, (doc, eg) in enumerate(nlp.pipe(texts, as_tuples=True, batch_size=10)):
        spans = []
        for ent in doc.ents:
            label = ent.label_
            if not labels or label in labels:
                start = ent.start_char
                end = ent.end_char
                spans.append({"start": start, "end": end, "label": label})
        task = {
            "id": i,
            "input": {"text": eg["text"]},
            "output": {"text": eg["text"], "spans": spans},
        }
        # Set the hashes for the newly created task
        task = set_hashes(task)
        yield task

def get_compare_questions(a_questions, b_questions):
    """Generate evaluation stream that consists of choice type tasks."""
    a_questions = {a["id"]: a for a in a_questions}
    b_questions = {b["id"]: b for b in b_questions}
    for id_, a in a_questions.items():
        # Ignore the questions that do not appear in both streams.
        if id_ not in b_questions:
            continue
        question = {
            **a["input"],
            "id": id_,
            "A": a["output"],
            "B": b_questions[id_]["output"],
        }
        # Ignore if the answers from both models are the same.
        if question["A"] == question["B"]:
            continue
        # Randomize the order of the outputs of the compared models.
        if  random.random() >= 0.5:
            order = ["B", "A"]
        else:
            order = ["A", "B"]
        # Add options for the choice interface.
        question["options"] = []
        for model_id in order:
            option = question[model_id]
            option["id"] = model_id
            question["options"].append(option)
        yield question

def print_results(ctrl):
    """Print the results of the evaluation to stout."""
    # Set the mapping from stream identifiers used in the tasks to meanigful stream names
    # to be used in the report.
    streamnames = {"A": "Before", "B": "After"}
    examples = ctrl.db.get_dataset(ctrl.dataset)
    counts = Counter()
    answers = {}
    for eg in examples:
        if "answer" not in eg:
            continue
        if "options" in eg:  # task created with choice UI
            selected = eg.get("accept", [])
            if not selected or len(selected) != 1 or eg["answer"] != "accept":
                continue
            answers[eg["id"]] = (eg["answer"], selected[0])
    for answer, selected in answers.values():
        if answer == "ignore":
            counts["ignore"] += 1
        else:
            counts[selected] += 1
    if not counts:
        raise ValueError("No answers found!")

    print("Evaluation results")
    pref, _ = counts.most_common(1)[0]
    if counts["A"] == counts["B"]:
        print("You had no preference")
        pref = None
    else:
        print(f"You preferred {pref} ({streamnames.get(pref)})")
    rows = [
        ("A", counts["A"], streamnames.get("A")),
        ("B", counts["B"], streamnames.get("B")),
        ("Ignored", counts["ignore"], ""),
        ("Total", sum(counts.values()), ""),
    ]
    print(tabulate(rows))


# Recipe decorator with argument annotations: (description, argument type,
# shortcut, type / converter function called on value before it's passed to
# the function). Descriptions are also shown when typing --help.
@prodigy.recipe(
    "ner.eval-ab",
    dataset=("The dataset to use", "positional", None, str),
    before_model=("Loadable spaCy pipeline with an entity recognizer", "positional", None, str),
    after_model=("Loadable spaCy pipeline with an entity recognizer", "positional", None, str),
    source=("The source data as a JSONL file", "positional", None, str),
    label=("One or more comma-separated labels", "option", "l", split_string),
    exclude=("Names of datasets to exclude", "option", "e", split_string),
    unsegmented=("Don't split sentences", "flag", "U", bool),
)
def ner_eval_ab(
    dataset: str,
    before_model: str,
    after_model: str,
    source: str,
    label: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    unsegmented: bool = False,
):
    """
    Evaluate two NER models by comparing their predictions and building an evaluation set from the stream.
    """

    before_nlp = spacy.load(before_model)
    after_nlp = spacy.load(after_model)

    # Load the stream from a JSONL file and return a generator that yields a
    # dictionary for each example in the data.
    stream = JSONL(source)

    if not unsegmented:
        # Use spaCy to split text into sentences
        stream = list(split_sentences(before_nlp, stream))

    # Generate tasks for both streams with the predictions of the models.
    before_stream = list(make_tasks(before_nlp, label, stream))
    after_stream = list(make_tasks(after_nlp, label, stream))
   
    # Generate choice tasks with models' predictions as options.
    stream = get_compare_questions(before_stream, after_stream)


    return {
        "view_id": "choice", # Annotation interface to use
        "dataset": dataset, # Name of dataset to save evaluation set
        "stream": stream, # Incoming stream of examples
        "on_exit": print_results,
        # Action to perform when the user stops the server. Here: print the evaluation results to stdout
        "exclude": exclude, # List of dataset names to exclude
        "config": {"auto_count_stream": True}, # Whether to recount the stream at initialization
    }