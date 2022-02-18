from collections import Counter
from typing import List, Optional
import random
import murmurhash
from tabulate import tabulate
import spacy
import prodigy
from prodigy.components.loaders import JSONL
from prodigy.util import split_string
from prodigy.components.preprocess import split_sentences


def make_tasks(nlp, labels, stream, name):
    """
        Generate a task for each example in a stream so that contains:
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
        # Set the hashes for the newly created task so as to differentiate
        # between `before` and `after` streams. Since `before` and `after` are not keys
        # in the task dictionary the hashes need to be set from scratch
        # (rather than using Prodigy `set_hashes` utility).
        task["_input_hash"] = murmurhash.hash(name + str(i))
        task["_task_hash"] = murmurhash.hash(name + str(i))
        yield task

def get_compare_questions(a_questions, b_questions):
    """Generate evaluation stream that consists of choice type tasks"""
    a_questions = {a["id"]: a for a in a_questions}
    b_questions = {b["id"]: b for b in b_questions}
    for id_, a in a_questions.items():
        # Ignore the questions that do not appear in both streams
        if id_ not in b_questions:
            continue
        question = {
            **a["input"],
            "id": id_,
            "A": a["output"],
            "B": b_questions[id_]["output"],
        }
        # Ignore if the answers from both models are the same
        if question["A"] == question["B"]:
            continue
        # Randomize the outputs of the compared models
        if  random.random() >= 0.5:
            question["mapping"] = {"B": "accept", "A": "reject"}
        else:
            question["mapping"] = {"A": "accept", "B": "reject"}
        # Add options for choice interface
        question["options"] = []
        for key in question["mapping"]:
            option = question[key]
            option["id"] = key
            question["options"].append(option)
        yield question
    
def print_results(ctrl):
    """Print the results of the evaluation to stout"""
    streamnames = {"A": "Before", "B": "After"}
    examples = ctrl.db.get_dataset(ctrl.dataset)
    counts = Counter()
    answers = {}
    # Get last example per ID
    for eg in examples:
        if "answer" not in eg or "mapping" not in eg:
            continue
        if "reject" not in eg:  # task created with choice UI
            selected = eg.get("accept", [])
            if not selected or len(selected) != 1 or eg["answer"] != "accept":
                continue
            eg["answer"] = eg["mapping"].get(selected[0])
        answers[eg["id"]] = (eg["answer"], eg["mapping"])
    for answer, mapping in answers.values():
        if answer == "ignore":
            counts["ignore"] += 1
        else:
            inverse = {v: k for k, v in mapping.items()}
            answer = inverse[answer]
            counts[answer] += 1
    if not counts:
        raise ValueError("No answers found")
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
    Evaluate two NER models by comparing their predictions and building an evaluation set from a stream.
    """

    before_nlp = spacy.load(before_model)
    after_nlp = spacy.load(after_model)

    # Load the stream from a JSONL file and convert the resulting generator to a list of
    # dictionaries of examples.
    # In this recipe we need to work with entire streams as lists to be able to generate comparison tasks.
    stream = list(JSONL(source))
    if not unsegmented:
        # Use spaCy to split text into sentences
        stream = list(split_sentences(before_nlp, stream))
    
    before_stream = list(make_tasks(before_nlp, label, stream, "before"))
    after_stream = list(make_tasks(after_nlp, label, stream, "after"))
    
    stream = list(get_compare_questions(before_stream, after_stream))


    return {
        "view_id": "choice", # Annotation interface to use
        "dataset": dataset, # Name of dataset to save evaluation set
        "stream": stream, # Incoming stream of examples
        "on_exit": print_results,
        # Action to perform when the user stops the server. Here: print the evaluation results to stdout
        "exclude": exclude, # List of dataset names to exclude
        "config": {"auto_count_stream": True}, # Whether to recount the stream at initialization 
    }