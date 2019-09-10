# Phrases

This recipe aims to take `terms.teach` one step further, allowing you to specify multiple words in phrases and get similar phrases using sense2vec

## Usage

Install Prodi.gy

Once Prodigy is installed, you should be able to run the `prodigy` command from
your terminal, either directly or via `python -m`:

Make sure to set API_URL to a deployed instance of the `spacy-services` example for sense2vec.
The code for that service is here: https://github.com/explosion/tree/master/sense2vec

If you want to quickly deploy this service I'd suggest using [Render](https://render.com).
If you find yourself using this recipe a lot you should definitely deploy your own version.

so...

```python
...
API_URL = "YOUR_URL"
...
```

But the real point of this recipe and the `terms.teach` recipe is to bootstrap a list of patterns as quickly as possible to start training an NER model. So (for **TESTING ONLY**) you can use the deployed version of the sense2vec powering the demo site. 

Note: there's no guaratee this api will work or always be up so please keep that in mind

```python
...
API_URL = "https://api.explosion.ai/sense2vec/find"
...
```


After you setup the `API_URL` you can run this recipe the same way you'd run terms.teach:

1. ### Create your dataset

```bash
prodigy dataset shoe_brands
```
2. ### Run the phrases.teach recipe


```bash
prodigy phrases.teach shoe_brands --seeds "Jordans,New Balance,Steve Madden" -F prodigy-recipes/contrib/phrases/phrases.py
```

You can also specify optional arguments:

* `-threshold`, `-t` (e.g. `-t 0.9`)
    This is the threshold for sense2vec similarity scores
* `-batch_size`, `-bs` (e.g. `-t 3`)
    The batch_size for this recipe. If you get caught in an infinite loop where use see Loading... forever, lower this batch size and you should be good.

    > NOTE: if you still see this Loading screen forever, your seed phrases probably don't exist in sense2vec


3. ### After training your phrase list, you'll want to export some patterns for training your NER or Text Classification model.


```bash
prodigy phrases.to-patterns shoe_brands ./shoe_brands_patterns.jsonl -F prodigy-recipes/contrib/phrases/phrases.py
```

And you should end up with a file that looks like:

```json
{"label": "SHOE_BRAND", "pattern": [{"LOWER": "jordans"}]}
{"label": "SHOE_BRAND", "pattern": [{"LOWER": "new"}, {"LOWER": "balance"}]}
{"label": "SHOE_BRAND", "pattern": [{"LOWER": "steve"}, {"LOWER": "madden"}]}
...
the rest of your patterns
```
---
### Now can use this patterns file with multi-word phrases to train multi-word Entities!
