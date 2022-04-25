from clumper import Clumper

import prodigy
from prodigy.components.db import connect


IMAGE_IDX = Clumper.read_jsonl("data/eifel_tower_indices.jsonl").map(lambda d: d['idx']).collect()
IMG_URL = "http://0.0.0.0:8000/generate"


@prodigy.recipe("simple-image")
def captcha_classification(dataset):
    def get_stream():
        for idx in IMAGE_IDX:
            yield {
                "id": idx,
                "html": f"<img src='{IMG_URL}/{idx}'/>", 
                "label": "eifel tower"
            }

    return {
        "dataset": dataset,
        "view_id": "classification",
        "stream": get_stream(),
        "config": {}
    }




from prodigy.components.db import connect

def fetch_existing_hashes(dataset):
    db = connect()
    existing_hashes = []
    if db.get_dataset(dataset):
        existing_hashes = [opt['_input_hash'] for d in db.get_dataset(dataset) 
                                              for opt in d['options']]
    return existing_hashes


@prodigy.recipe("captcha")
def captcha_classification(dataset,):
    def get_stream():
        existing_hashes = fetch_existing_hashes(dataset)
        options = []
        for i in IMAGE_IDX:
            option = {"id": i, "image": f"{IMG_URL}/{i}"}
            option = prodigy.set_hashes(option, input_keys=("id",))
            if option["_input_hash"] not in existing_hashes:
                options.append(option)
            if len(options) == 9:
                yield {"options": options, "text": "Please select Eiffel Towers."}
                options = []

    return {
        "dataset": dataset,
        "view_id": "choice",
        "stream": get_stream(),
        "config": {"choice_style": "multiple"}
    }
