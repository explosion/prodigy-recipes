# dedupe.io with Prodi.gy

This is a custom recipe for linking records across multiple datasets using the Python [dedupe](https://github.com/dedupeio/dedupe) library.
See https://github.com/dedupeio/dedupe-examples/tree/master/record_linkage_example for an example of linking records with dedupe's console labeler to compare.

## Usage

Install Prodi.gy

Once Prodigy is installed, you should be able to run the `prodigy` command from
your terminal, either directly or via `python -m`:

Install Requirements
```bash
pip install -r requirements.txt
```

Run with example datasets
```bash
python -m prodigy records.link my_dataset --left data/raw_dedupe_abtbuy_abt.csv --right data/raw_dedupe_abtbuy_buy.csv --fields fields.json -F ./link_records.py
```

---
### Annotating

![annotation interface](img/link_records_example.jpg)
In the interface, a row is highlighted green if the field has an exact string match across both datasets, otherwise the row will be green.

If you think the records are duplicates like they are in the image above, accept, otherwise reject.

When you click the save button your progress will be updated.

In order to reach 100% progress, the dedupe library recommends at least 10 positive and 10 negative examples.

---
### Model training
Once you end the annotation session, a model will be batch trained and evaluated on the rest of your dataset and will write out records the model think should be conflated together to a file named `data_matching_output.csv` and save a copy of the dedupe model settings to `data_matching_learned_settings`
