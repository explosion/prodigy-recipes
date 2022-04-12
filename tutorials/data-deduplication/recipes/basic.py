import prodigy
from prodigy.components.loaders import JSONL
from jinja2 import Environment, select_autoescape, FileSystemLoader, Markup


@prodigy.recipe(
    "duplicate",
    dataset=("The dataset to save to", "positional", None, str),
    file_path=("The jsonl file with matched items", "positional", None, str),
)
def check_duplicate(dataset, file_path):
    """Annotate yes/no duplicate."""
    stream = JSONL(file_path)     # load in the JSONL file
    stream = add_options(stream)  # add options to each task

    return {
        "dataset": dataset,   # save annotations in this dataset
        "view_id": "choice",  # use the choice interface
        "stream": stream
    }

def add_options(stream):
    env = Environment(
        loader=FileSystemLoader('recipes'),
        autoescape=select_autoescape(['html', 'xml'])
    )
    # Helper function to add options to every task in a stream
    options = [
        {"id": "duplicate", "text": "✅ duplicate"},
        {"id": "unique", "text": "❌ unique"},
    ]
    for task in stream:
        task["options"] = options
        task["html"] = env.get_template("basic.html").render(item1=task['item1'], item2=task['item2'])
        yield task
