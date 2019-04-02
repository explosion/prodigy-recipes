# coding: utf8
from __future__ import unicode_literals

import prodigy
from prodigy.components.loaders import JSONL


# Recipe decorator with argument annotations: (description, argument type,
# shortcut, type / converter function called on value before it's passed to
# the function). Descriptions are also shown when typing --help.
@prodigy.recipe(
    "question-answering",
    dataset=("The dataset to use", "positional", None, str),
    source=("The source data as a JSONL file", "positional", None, str),
)
def question_answering(dataset, source):
    """
    Annotate question/answer pairs with a custom HTML interface. Expects an
    input file with records that look like this:

        {"question": "What color is the sky?", "question_answer": "blue"}

    Important note: The "answer" field is reserved by Prodigy and will be set
    in the annotation UI ("accept", "reject" or "ignore"). That's why we're
    using "question_answer" here.
    """
    # Load the stream from a JSONL file and return a generator that yields a
    # dictionary for each example in the data.
    stream = JSONL(source)

    # The HTML template to use. While we could also reformat the stream to
    # include a "html" field for each example, a template allows rendering
    # tasks without having to include the full HTML markup every time. All
    # task properties become available as Mustache-style variables.
    html_template = (
        "<div style='text-align: left; width: 100%'>"
        "<div style='padding: 20px; border-bottom: 1px solid #ccc'><strong>Question:</strong> {{question}}</div>"
        "<div style='padding: 20px'><strong>Answer:</strong> {{question_answer}}</div>"
        "</div>"
    )

    return {
        "view_id": "html",  # Annotation interface to use
        "dataset": dataset,  # Name of dataset to save annotations
        "stream": stream,  # Incoming stream of examples
        "config": {"html_template": html_template},  # Additional config
    }
