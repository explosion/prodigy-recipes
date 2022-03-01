import copy
from typing import List, Optional
import spacy
from spacy.training import Example
import prodigy
from prodigy.components.loaders import JSONL
from prodigy.components.preprocess import add_tokens, split_sentences
from prodigy.util import split_string, set_hashes

def make_tasks(nlp, stream, labels):
    """Add a 'spans' key to each example, with predicted entities."""
    # Process the stream using spaCy's nlp.pipe, which yields doc objects.
    # If as_tuples=True is set, you can pass in (text, context) tuples.
    texts = ((eg["text"], eg) for eg in stream)
    for doc, eg in nlp.pipe(texts, as_tuples=True):
        task = copy.deepcopy(eg)
        spans = []
        for ent in doc.ents:
            # Ignore if the predicted entity is not in the selected labels.
            if labels and ent.label_ not in labels:
                continue
            # Create a span dict for the predicted entity.
            spans.append(
                {
                    "token_start": ent.start,
                    "token_end": ent.end - 1,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "text": ent.text,
                    "label": ent.label_,
                }
            )
        task["spans"] = spans
        # Rehash the newly created task so that hashes reflect added data.
        task = set_hashes(task)
        yield task


# Recipe decorator with argument annotations: (description, argument type,
# shortcut, type / converter function called on value before it's passed to
# the function). Descriptions are also shown when typing --help.
@prodigy.recipe(
    "ner.correct",
    dataset=("The dataset to use", "positional", None, str),
    spacy_model=("The base model", "positional", None, str),
    source=("The source data as a JSONL file", "positional", None, str),
    label=("One or more comma-separated labels", "option", "l", split_string),
    update=("Whether to update the model during annotation", "flag", "UP", bool),
    exclude=("Names of datasets to exclude", "option", "e", split_string),
    unsegmented=("Don't split sentences", "flag", "U", bool),
    component=("Name of NER component in the pipeline", "option", "c", str),
)
def ner_correct(
    dataset: str,
    spacy_model: str,
    source: str,
    label: Optional[List[str]] = None,
    update: bool = False,
    exclude: Optional[List[str]] = None,
    unsegmented: bool = False,
    component: Optional[str] = "ner",
):
    """
    Create gold-standard data by correcting a model's predictions manually.
    This recipe used to be called `ner.make-gold`.
    """
    # Load the spaCy model.
    nlp = spacy.load(spacy_model)

    labels = label

    # Get existing model labels, if available.
    if component not in nlp.pipe_names:
        raise ValueError(f"Can't find component '{component}' in the provided pipeline.")
    model_labels = nlp.pipe_labels.get(component, [])

    # Check if we're annotating all labels present in the model or a subset.
    use_all_model_labels = len(set(labels).intersection(set(model_labels))) == len(model_labels)

    # Load the stream from a JSONL file and return a generator that yields a
    # dictionary for each example in the data.
    stream = JSONL(source)

    if not unsegmented:
        # Use spaCy to split text into sentences.
        stream = split_sentences(nlp, stream)

    # Tokenize the incoming examples and add a "tokens" property to each
    # example. Also handles pre-defined selected spans. Tokenization allows
    # faster highlighting, because the selection can "snap" to token boundaries.
    stream = add_tokens(nlp, stream)

    # Add the entities predicted by the model to the tasks in the stream.
    stream = make_tasks(nlp, stream, labels)

    def make_update(answers):
        """Update the model with the received answers to improve future suggestions"""
        examples = []
        # Set the default label for the tokens outside the provided spans.
        default_label = "outside" if use_all_model_labels else "missing"
        for eg in answers:
            if eg["answer"] == "accept":
                # Create a "predicted" doc object and a "reference" doc objects to be used
                # as a training example in the model update. If your examples contain tokenization
                # make sure not to loose this information by initializing the doc object from scratch.
                pred = nlp.make_doc(eg["text"])
                ref = nlp.make_doc(eg["text"])
                spans = [
                    pred.char_span(span["start"], span["end"], label=span["label"])
                    for span in eg.get("spans", [])
                ]
                # Use the information in spans to set named entites in the document specifying
                # how to handle the tokens outside the provided spans.
                ref.set_ents(spans, default=default_label)
                examples.append(Example(pred, ref))
        nlp.update(examples)

    return {
        "view_id": "ner_manual",  # Annotation interface to use
        "dataset": dataset,  # Name of dataset to save annotations
        "stream": stream,  # Incoming stream of examples
        "update": make_update if update else None, # Update the model in the loop if required
        "exclude": exclude,  # List of dataset names to exclude
        "config": {  # Additional config settings, mostly for app UI
            "lang": nlp.lang,
            "labels": labels,  # Selectable label options
            "exclude_by": "input", # Hash value to filter out seen examples
            "auto_count_stream": not update, # Whether to recount the stream at initialization
        },
    }