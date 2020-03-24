import prodigy
from prodigy.components.loaders import Images
from prodigy.components.db import connect
from collections import Counter

from image_captioning_model import load_model, generate_caption


@prodigy.recipe("image-caption")
def image_caption(dataset, images_path):
    """Stream in images from a directory and allow captioning them by typing
    a caption in a text field. The caption is stored as the key "caption".
    """
    stream = Images(images_path)
    blocks = [
        {"view_id": "image"},
        {"view_id": "text_input", "field_id": "caption", "field_autofocus": True},
    ]
    return {
        "dataset": dataset,
        "stream": stream,
        "view_id": "blocks",
        "config": {"blocks": blocks},
    }


@prodigy.recipe("image-caption.correct")
def image_caption_correct(dataset, images_path):
    """Stream in images from a directory and pre-populate the text box with the
    model's predicted caption. The original caption is stored as "orig_caption",
    the potentially edited caption as "caption".
    """
    encoder, decoder, vocab, transform = load_model()
    blocks = [
        {"view_id": "image"},
        {"view_id": "text_input", "field_id": "caption", "field_autofocus": True},
    ]
    counts = {"changed": 0, "unchanged": 0}

    def get_stream():
        stream = Images(images_path)
        for eg in stream:
            caption = generate_caption(eg["image"], encoder, decoder, vocab, transform)
            eg["caption"] = caption
            eg["orig_caption"] = caption
            yield eg

    def update(answers):
        for eg in answers:
            if eg["answer"] == "accept":
                if eg["caption"] != eg["orig_caption"]:
                    counts["changed"] += 1
                else:
                    counts["unchanged"] += 1

    def on_exit(ctrl):
        print("\nResults")
        print(counts["changed"], "changed")
        print(counts["unchanged"], "unchanged")

    return {
        "dataset": dataset,
        "stream": get_stream(),
        "update": update,
        "on_exit": on_exit,
        "view_id": "blocks",
        "config": {"blocks": blocks},
    }


@prodigy.recipe("image-caption.diff")
def image_caption_diff(dataset, source_dataset):
    """Review an existing captions dataset collected with image-caption.correct
    and go through all captions that were edited. Display both and annotate
    why the caption was changed using multiple choice options.
    """
    db = connect()
    examples = db.get_dataset(source_dataset)
    counts = Counter()

    blocks = [
        {
            "view_id": "html",
            "html_template": "<div style='opacity: 0.5'>{{orig_caption}}</div>",
        },
        {"view_id": "html", "html_template": "{{caption}}"},
        {"view_id": "choice"},
    ]

    options = [
        {"id": "SUBJECT", "text": "🐶 wrong subject"},
        {"id": "ATTRS", "text": "🎨 wrong subject attributes"},
        {"id": "BACKGROUND", "text": "🖼 wrong background or setting"},
        {"id": "NUMBER", "text": "🧮 wrong number"},
        {"id": "WORDING", "text": "💬 wording or spelling change"},
        {"id": "OTHER", "text": "🤷‍♀️ other mistakes"},
    ]

    def get_stream():
        for eg in examples:
            if eg["answer"] == "accept" and eg["caption"] != eg["orig_caption"]:
                eg["options"] = options
                yield eg

    def update(answers):
        for eg in answers:
            if eg["answer"] == "accept":
                selected = eg.get("accept", [])
                for opt_id in selected:
                    counts[opt_id] += 1

    def on_exit(ctrl):
        print("\nMistakes")
        for opt_id, i in counts.items():
            print(i, opt_id)

    return {
        "dataset": dataset,
        "stream": get_stream(),
        "update": update,
        "on_exit": on_exit,
        "view_id": "blocks",
        "config": {"blocks": blocks, "choice_style": "multiple"},
    }
