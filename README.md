<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# Prodigy Recipes

This repository contains a collection of recipes for [Prodigy](https://prodi.gy),
our scriptable annotation tool for text, images and other data. In order to use
this repo, you'll need a license for Prodigy –
[see this page](https://prodi.gy/buy) for more details. For questions and bug
reports, please use the [Prodigy Support Forum](https://support.prodi.gy). If
you've found a mistake or bug, feel free to submit a
[pull request](https://github.com/explosion/prodigy-recipes/pulls).

> ✨ **Important note:** The recipes in this repository aren't 100% identical to
> the built-in recipes shipped with Prodigy. They've been edited to include
> comments and more information, and some of them have been simplified to make
> it easier to follow what's going on, and to use them as the basis for a
> custom recipe.

## 📋 Usage

Once Prodigy is installed, you should be able to run the `prodigy` command from
your terminal, either directly or via `python -m`:

```bash
python -m prodigy
```

The `prodigy` command lists the built-in recipes. To use a custom recipe script,
simply pass the path to the file using the `-F` argument:

```bash
python -m prodigy -F prodigy-recipes/ner/ner_teach.py
```

Now you can edit the code in your `ner_teach.py` command, to customize how
Prodigy behaves.

### Some things to try

* Try replacing `prefer_uncertain()` with `prefer_high_scores()`.
* Try writing a custom sorting function. It just needs to be a generator that
  yields a sequence of `example` dicts, given a sequence of `(score, example)` tuples.
* Try adding a filter that drops some questions from the stream. For instance,
  try writing a filter that only asks you questions where the entity is two
  words long.
* Try customizing the `update()` callback, to include extra logging or extra
  functionality.

## 🍳 Recipes

### Named Entity Recognition

| Recipe | Description |
| --- | --- |
| [`ner.teach`](ner/ner_teach.py) | Collect the best possible training data for a named entity recognition model with the model in the loop. Based on your annotations, Prodigy will decide which questions to ask next. |
| [`ner.match`](ner/ner_match.py) | Suggest phrases that match a given patterns file, and mark whether they are examples of the entity you're interested in. The patterns file can include exact strings or token patterns for use with spaCy's `Matcher`. |
| [`ner.manual`](ner/ner_manual.py) | Mark spans manually by token. Requires only a tokenizer and no entity recognizer, and doesn't do any active learning. |
| [`ner.make-gold`](ner/ner_make-gold.py) | Create gold-standard data by correcting a model's predictions manually. |
| [`ner.silver-to-gold`](ner/ner_silver_to_gold.py) | Take an existing "silver" dataset with binary accept/reject annotations, merge the annotations to find the best possible analysis given the constraints defined in the annotations, and manually edit it to create a perfect and complete "gold" dataset. |

### Text Classification

| Recipe | Description |
| --- | --- |
| [`textcat.teach`](textcat/textcat_teach.py) | Collect the best possible training data for a text classification model with the model in the loop. Based on your annotations, Prodigy will decide which questions to ask next. |
| [`textcat.custom-model`](textcat/textcat_custom_model.py) | Use active learning-powered text classification with a custom model. To demonstrate how it works, this demo recipe uses a simple dummy model that "predicts" random scores. But you can swap it out for any model of your choice, for example a text classification model implementation using PyTorch, TensorFlow or scikit-learn. |

### Terminology

| Recipe | Description |
| --- | --- |
| [`terms.teach`](terms/terms_teach.py) | Bootstrap a terminology list with word vectors and seeds terms. Prodigy will suggest similar terms based on the word vectors, and update the target vector accordingly. |

### Image

| Recipe | Description |
| --- | --- |
| [`image.manual`](image/image_manual.py) | Manually annotate images by drawing rectangular bounding boxes or polygon shapes on the image. |

### Other

| Recipe | Description |
| --- | --- |
| [`mark`](other/mark.py) | Click through pre-prepared examples, with no model in the loop. |
| [`choice`](other/choice.py) | Annotate data with multiple-choice options. The annotated examples will have an additional property `"accept": []` mapping to the ID(s) of the selected option(s). |

## 📚 Example Datasets and Patterns

To make it even easier to get started, we've also included a few
[`example-datasets`](example-datasets), both raw data as well as data containing
annotations created with Prodigy. For examples of token-based match patterns to
use with recipes like `ner.teach` or `ner.match`, see the
[`example-patterns`](example-patterns) directory.
