# Bulk Labelling Images with Finetuning via Prodigy

This project contains a small project that demonstrates how you
might set up [Prodi.gy](https://prodi.gy) to do bulk labelling 
for image classification. 

![](bulk.png)

This tutorial relies on the [bulk library](https://github.com/koaning/bulk) for
quick selections and it uses Prodigy to help fine-tune the embeddings. 

![](emb.webp)

If you're curious, this project is the end-result of [this Youtube tutorial](https://youtu.be/DmH3JmX3w2I). Check it out if you want to learn more about pretrained models and finetuning!

## Installation 

The installation process is automated via `make`.

```
python -m pip install -r requirements.txt
```

In order for the install to work, you'll need to
add a `.env` file to the root of the project that
contains your prodigy license key. It should look
something like:

```
PRODIGY_KEY="1234-ABCD-5678-EFGH"
```

The tutorial video also uses a few datasets that you can download via: 

```
python -m bulk download twemoji
python -m bulk download pets
```

## Contents 

The project has a few notable files/folders: 

- The `make_emoji.py`/`make_pets.py` scripts are examples of how to generate .csv files for the bulk annotation interface.
- The `finetune.ipynb` notebook contains code that can finetune the embeddings given a few annotated examples.
