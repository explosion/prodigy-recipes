import copy
from typing import List, Optional
import prodigy
from prodigy.components.loaders import JSONL
from prodigy.util import split_string
import spacy
from spacy.tokens import Doc
from spacy.training import Example


# Recipe decorator with argument annotations: (description, argument type,
# shortcut, type / converter function called on value before it's passed to
# the function). Descriptions are also shown when typing --help.
@prodigy.recipe(
    "textcat.correct",
    dataset=("The dataset to use", "positional", None, str),
    spacy_model=("The base model", "positional", None, str),
    source=("The source data as a JSONL file", "positional", None, str),
    label=("One or more comma-separated labels", "option", "l", split_string),
    update=("Whether to update the model during annotation", "flag", "UP", bool),
    exclude=("Names of datasets to exclude", "option", "e", split_string),
    threshold=("Score threshold to pre-select label", "option", "t", float),
    component=("Name of text classifier component in the pipeline (will be guessed from pipeline if not set)", "option", "c", str),
)

def textcat_correct(
    dataset: str,
    spacy_model: str,
    source: str,
    label: Optional[List[str]] = None,
    update: bool = False,
    exclude: Optional[List[str]] = None,
    threshold: float = 0.5,
    component: Optional[str] = None,
):
    """
    Correct the textcat model's predictions manually. Only the predictions
    above the threshold will be pre-selected. By default, all labels with a score 0.5 and above will
    be accepted automatically. In the built-in "textcat.correct" recipe Prodigy would infer whether
    the categories should be mutualy exclusive based on the component configuration.
    Here, for demo purposes, we show how it can be inferred from the pipeline config.
    """
    # Load the stream from a JSONL file and return a generator that yields a
    # dictionary for each example in the data.
    stream = JSONL(source)

    # Load the spaCy model
    nlp = spacy.load(spacy_model)

    # Get a valid classifier component from pipeline
    if not component:
        component = "textcat" if "textcat" in nlp.pipe_names else "textcat_multilabel"

    # Infer whether the labels are exclusive from pipeline config
    pipe_config = nlp.get_pipe_config(component)
    exclusive = pipe_config.get("model", {}).get("exclusive_classes", True)

    # Get labels from the model in case they are not provided
    labels = label
    if not labels:
        labels = nlp.pipe_labels.get(component, [])
    
    # Add classifier predictions to each task in stream under 'options' key with the score per category
    # and 'selected' key with the categories above the threshold.
    def add_suggestions(stream):
        texts = ((eg["text"], eg) for eg in stream)
        # Process the stream using spaCy's nlp.pipe, which yields doc objects.
        # If as_tuples=True is set, you can pass in (text, context) tuples.
        for doc, eg in nlp.pipe(texts, as_tuples=True, batch_size=10):
            task = copy.deepcopy(eg)
            options = []
            selected = []
            for cat, score in doc.cats.items():
                if cat in labels:
                    options.append({"id": cat, "text": cat, "meta": f"{score:.2f}"})
                    if score >= threshold:
                        selected.append(cat)
            task["options"] = options
            task["accept"] = selected
            yield task

    # Update the model with the corrected examples.
    def make_update(answers):
        examples=[]
        for eg in answers:
            if eg["answer"] == "accept":
                selected = eg.get("accept", [])
                cats = {
                    opt["id"]: 1.0 if opt["id"] in selected else 0.0
                    for opt in eg.get("options", [])
                }
                # Create a doc object to be used as a training example in the model update.
                # If your examples contain tokenization make sure not to loose this information
                # by initializing a doc object from scratch.
                doc = nlp.make_doc(eg["text"])
                examples.append(Example.from_dict(doc, {"cats": cats}))
        nlp.update(examples)

    # Add model's predictions to the tasks in the stream.
    stream = add_suggestions(stream)

    return {
        "view_id": "choice", # Annotation interface to use
        "dataset": dataset,  # Name of dataset to save annotations
        "stream": stream,  # Incoming stream of examples
        "update": make_update if update else None,
        "exclude": exclude,  # List of dataset names to exclude
        "config": {  # Additional config settings, mostly for app UI
            # Style of choice interface
            "choice_style": "single" if exclusive and len(label) > 1 else "multiple",
            "exclude_by": "input", # Hash value to filter out seen examples
            "auto_count_stream": not update, # Whether to recount the stream at initialization 
        },
    }
