# Manual and model-assisted image captioning

<a href="https://www.youtube.com/watch?v=zlyq9z7hdUA" target="_blank"><img src="https://user-images.githubusercontent.com/13643239/77343162-8f1ba480-6d31-11ea-9145-e0e5c44ba44b.png" width="300" height="auto" align="right" /></a>

This directory contains a recipe scripts for collecting and reviewing image
captioning data with [Prodigy](https://prodi.gy). The captioning model is
implemented in PyTorch based on
[this tutorial](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning).
To use the pretrained model, download the files
[from here](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning#pretrained-model)
and place them all in this directory. For more details on custom recipes with
Prodigy, check out [the documentation](https://prodi.gy/docs/custom-recipes).

> 📺 **This project was created as part of a
> [step-by-step video tutorial](https://www.youtube.com/watch?v=zlyq9z7hdUA).**

## Usage

For more details on the recipes, check out
[`image_caption.py`](image_caption.py) or run a recipe with `--help`, for
example: `prodigy image-caption -F image_caption.py --help`.

### <kbd>recipe</kbd> `image-caption`: Collect image captions manually

Start the server, stream in images from a directory and allow annotating them
with captions. Captions will be saved in the data as the field `"caption"`.

```bash
prodigy image-caption caption_data ./images -F image_caption.py
```

### <kbd>recipe</kbd> `image-caption.correct`: Model-assisted image captioning

Start the server, stream in images from a directory and display the generated
captions in the text field, allowing the annotator to change them if needed.
Captions will be saved in the data as the field `"caption"` and the original
unedited caption will be preserved as `"orig_caption"`. Prints the counts of
changed vs. unchanged captions on exit.

```bash
prodigy image-caption.correct caption_data ./images -F image_caption.py
```

This recipe expects the files `vocab.pkl`, `encoder-5-3000.pkl` and
`decoder-5-3000.pkl` to be present in the same directory. You can download a
pretrained model
[from here](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning#pretrained-model).
If needed, the recipe could be edited to allow the model path to be passed in as
a [recipe argument](https://prodi.gy/docs/custom-recipes#recipe-args) that's
then passed to `load_model`.

### <kbd>recipe</kbd> `image-caption.diff`: Review corrected image captions

Go through all edited captions in a dataset created with `image-caption.correct`
and select why the caption was changed, based on multiple choice options. Prints
the counts of options on exit.

```bash
prodigy image-caption.correct caption_data_diff caption_data -F image_caption.py
```

The options are currently hard-coded in the recipe
[`image_caption.py`](image_caption.py), but the recipe could be modified to take
a JSON file of options instead via a
[recipe argument](https://prodi.gy/docs/custom-recipes#recipe-args).
