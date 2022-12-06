import srsly 
import prodigy 
from prodigy.components.db import connect

@prodigy.recipe(
    "terms.from-ner",
    ner_dataset=("Dataset loader NER annotations from", "positional", None, str),
    file_out=("File to write patterns into", "positional", None, str)
)
def custom_recipe(ner_dataset: str, file_out: str):
    # Connect to Prodigy database
    db = connect()
    # Load in annotated examples 
    annotated = db.get_dataset(ner_dataset)
    # Loop over examples
    pattern_set = set()
    for example in annotated:
        for span in example.get("spans", []):
            pattern_str = example['text'][span['start']: span['end']]
            # Store into tuple, because sets like that
            tup = (pattern_str, span['label'])
            pattern_set.add(tup)
    patterns = [{"pattern": p, "label": l} for p, l in pattern_set]
    srsly.write_jsonl(file_out, patterns)
