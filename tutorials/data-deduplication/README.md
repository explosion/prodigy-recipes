# Prodigy Deduplication Demo 

This project contains a small project that demonstrates how you
might set up [Prodigy](https://prodi.gy) for a data deduplication
task. The project relies on the [recordlinkage](https://recordlinkage.readthedocs.io/en/latest/index.html) library for heuristics.

![preview](preview.png)

If you're curious, this project is the end-result of [this YouTube tutorial](https://www.youtube.com/watch?v=kJ5Jb56T5uc). Check it out if you want to learn more about Prodigy!

## Installation 

The installation process is automated via `make`.

```
make install
```

In order for the install to work, you'll need to
add a `.env` file to the root of the project that
contains your Prodigy license key. It should look
something like:

```
PRODIGY_KEY="1234-ABCD-5678-EFGH"
```

## Contents 

The project has a few notable files/folders: 

- The `recipes` folder contains Prodigy recipes along with some associated templates. These are used to generate an appropriate UI for the data deduplication task. 
- The `prodigy.json` file contains some settings for Prodigy. You can set the hostname and portnumber from here.
- The `data` folder contains `.jsonl` files with likely duplicate records. These have been generated using heuristics found in the `dedup.ipynb` notebook.

## Usage

You can run Prodigy via `make` too. 

```bash
# Run Prodigy with the basic recipe 
make prodigy-basic

# Run Prodigy with the improved recipe 
make prodigy-intermediate
```
