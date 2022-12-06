from typing import List, Optional, Union, Iterable
import prodigy 
from prodigy.components.loaders import get_stream
from prodigy.components.preprocess import fetch_media as fetch_media_preprocessor
from prodigy.util import log, msg, get_labels, split_string
from prodigy.types import TaskType, RecipeSettingsType


def remove_base64(examples: List[TaskType]) -> List[TaskType]:
    """Remove base64-encoded string if "path" is preserved in example."""
    for eg in examples:
        if "audio" in eg and eg["audio"].startswith("data:") and "path" in eg:
            eg["audio"] = eg["path"]
        if "video" in eg and eg["video"].startswith("data:") and "path" in eg:
            eg["video"] = eg["path"]
    return examples

@prodigy.recipe(
    "audio.manual-with-transcript",
    # fmt: off
    dataset=("Dataset to save annotations to", "positional", None, str),
    source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    loader=("Loader to use", "option", "lo", str),
    label=("Comma-separated label(s) to annotate or text file with one label per line", "option", "l", get_labels),
    keep_base64=("If 'audio' loader is used: don't remove base64-encoded data from the data on save", "flag", "B", bool),
    autoplay=("Autoplay audio when a new task loads", "flag", "A", bool),
    fetch_media=("Convert URLs and local paths to data URIs", "flag", "FM", bool),
    exclude=("Comma-separated list of dataset IDs whose annotations to exclude", "option", "e", split_string),
    # fmt: on
)
def custom(
    dataset: str,
    source: Union[str, Iterable[dict]],
    loader: Optional[str] = "audio",
    label: Optional[List[str]] = None,
    autoplay: bool = False,
    keep_base64: bool = False,
    fetch_media: bool = False,
    exclude: Optional[List[str]] = None,
) -> RecipeSettingsType:
    log("RECIPE: Starting recipe audio.custom", locals())
    if label is None:
        msg.fail("audio.custom requires at least one --label", exits=1)
    stream = get_stream(source, loader=loader, dedup=True, rehash=True, is_binary=False)
    if fetch_media:
        stream = fetch_media_preprocessor(stream, ["audio", "video"])
    
    blocks = [
        {"view_id": "audio_manual"},
        {
            "view_id": "text_input",
            "field_rows": 4,
            "field_label": "Transcript",
            "field_id": "transcript",
            "field_autofocus": True,
        },
    ]

    return {
        "view_id": "blocks",
        "dataset": dataset,
        "stream": stream,
        "before_db": remove_base64 if not keep_base64 else None,
        "exclude": exclude,
        "config": {
            "blocks": blocks,
            "labels": label,
            "audio_autoplay": autoplay,
            "auto_count_stream": True,
        },
    }
