# Prodigy Deduplication Demo 

This project contains a small project that demonstrates how you
might set up [prodigy](https://prodi.gy) for a data deduplication
task. The project relies on the [recordlinkage](https://recordlinkage.readthedocs.io/en/latest/index.html) library for heuristics.

![preview](preview.png)

If you're curious, this project is the end-result of [this Youtube tutorial](). Check it out if you want to learn more about prodigy!

## Installation 

The installation process is automated via `make`.

```
make install
```

In order for the install to work, you'll need to
add a `.env` file to the root of the project that
contains your prodigy license key. It should look
something like:

```
PRODIGY_KEY="1234-ABCD-5678-EFGH"
```

## Contents 

The project has a few notable files/folders: 

- The `recipes` folder contains prodigy recipes along with some associated templates. These are used to generate an appropriate UI for the data deduplication task. 
- The `prodigy.json` file contains some settings for prodigy. You can set the hostname and portnumber from here.
- The `data` folder contains `.jsonl` files with likely duplicate records. These have been generated using heuristics found in the `dedup.ipynb` notebook.

## Usage

You can run prodigy via `make` too. 

```bash
# Run prodigy with the basic recipe 
make prodigy-basic

# Run prodigy with the improved recipe 
make prodigy-intermediate
```