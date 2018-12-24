# coding: utf8
from __future__ import unicode_literals

import json
import csv
import re

import dedupe
from unidecode import unidecode
import prodigy
from prodigy.components.db import connect


def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def preProcess(column):
    """
    Do a little bit of data cleaning with the help of Unidecode and Regex.
    Things like casing, extra spaces, quotes and new lines can be ignored.
    """

    column = unidecode(column)
    column = re.sub('\n', ' ', column)
    column = re.sub('-', '', column)
    column = re.sub('/', ' ', column)
    column = re.sub("'", '', column)
    column = re.sub(",", '', column)
    column = re.sub(":", ' ', column)
    column = re.sub('  +', ' ', column)
    column = column.strip().strip('"').strip("'").lower().strip()
    if not column:
        column = None
    return column


def readData(filename):
    """
    Read in our data from a CSV file and create a dictionary of records,
    where the key is a unique record ID.
    """

    data_d = {}

    with open(filename) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            clean_row = dict([(k, preProcess(v)) for (k, v) in row.items()])
            if clean_row['price']:
                clean_row['price'] = float(clean_row['price'][1:])
            data_d[filename + str(i)] = dict(clean_row)

    return data_d


def record_pairs_stream(linker):  # pragma: no cover
    '''
    Command line interface for presenting and labeling training pairs
    by the user
    Argument :
    A deduper object
    '''

    finished = False
    use_previous = False
    fields = unique(
        field.field
        for field in linker.data_model.primary_fields
    )

    examples_buffer = []
    uncertain_pairs = []

    while not finished:
        if use_previous:
            record_pair, _ = examples_buffer.pop(0)
            use_previous = False
        else:
            if not uncertain_pairs:
                uncertain_pairs = linker.uncertainPairs()

            try:
                record_pair = uncertain_pairs.pop()
                a, b = record_pair
                stream = []

                for field_name in list(a.keys()):
                    if field_name in fields:
                        exact_match = a[field_name] == b[field_name]
                        stream.append({
                            'name': field_name,
                            'a_value': a[field_name],
                            'b_value': b[field_name],
                            'exact_match': exact_match,
                            'not_exact_match': not exact_match
                        })
                yield {'fields': stream}
            except IndexError:
                break


def update_linker(linker, examples):
    labeled_pairs = {'distinct': [], 'match': []}

    for e in examples:

        record_a = {}
        record_b = {}
        for field in e['fields']:
            record_a[field['name']] = field['a_value']
            record_b[field['name']] = field['b_value']

        record_pair = (record_a, record_b)

        if e['answer'] == 'accept':
            labeled_pairs['match'].append(record_pair)
        elif e['answer'] == 'reject':
            labeled_pairs['distinct'].append(record_pair)

    linker.markPairs(labeled_pairs)
    return linker


def validate_field(field):
    assert 'field' in field
    assert 'type' in field


# Recipe decorator with argument annotations: (description, argument type,
# shortcut, type / converter function called on value before it's passed to
# the function). Descriptions are also shown when typing --help.
@prodigy.recipe('records.link',
    dataset=("The dataset to use", "positional", None, str),
    left_record_file_path=("One of two files to dedupe and conflate across. Will be on the left in annotation UI", "positional", None, str),
    right_record_file_path=("One of two files to dedupe and conflate across. Will be on the right in annotation UI", "positional", None, str),
    fields_json_file_path=("The path to a JSON config file for field dedupe", "positional", None, str)
)
def link_records(dataset, left_record_file_path, right_record_file_path, fields_json_file_path):
    """
    Collect the best possible training data for linking records across multiple
    datasets. This recipe is an example of linking records across 2 CSV files
    using the dedupe.io library.
    """

    db = connect()  # uses the settings in your prodigy.json

    output_file = 'data_matching_output.csv'
    settings_file = 'data_matching_learned_settings'
    training_file = 'data_matching_training.json'

    left_records = readData(left_record_file_path)
    right_records = readData(right_record_file_path)

    def descriptions():
        for dataset in (left_records, right_records):
            for record in dataset.values():
                yield record['description']

    with open(fields_json_file_path) as fields_json_file:
        fields = json.load(fields_json_file)

    for field in fields:
        validate_field(field)
        if field['type'] == 'Text' and 'corpus' in field:            
            func_name = field['corpus'][1:-1]
            field['corpus'] = locals()[func_name].__call__()

    print('LEN RECORDS: ', len(left_records) / 2, len(right_records) / 2)
    print('MIN SAMPLE', min(len(left_records) / 2, len(right_records) / 2))
    print(fields)

    linker = dedupe.RecordLink(fields)
    # To train the linker, we feed it a sample of records.
    linker.sample(
        left_records,
        right_records,
        round(min(len(left_records) / 2, len(right_records) / 2))
    )

    print('getting examples')
    # If we have training data saved from a previous run of linker,
    # look for it an load it in.
    examples = db.get_dataset(dataset)
    if len(examples) > 0:
        linker = update_linker(linker, examples)

    def update(examples, linker=linker):
        print(len(examples))
        linker = update_linker(linker, examples)

    def get_progress(session=0, total=0, loss=0, linker=linker):
        n_match = len(linker.training_pairs['match'])
        n_distinct = len(linker.training_pairs['distinct'])
        n_match_progress = min(1, (n_match / 10)) / 2
        n_distinct_progress = min(1, (n_distinct / 10)) / 2
        print("Examples Annotated: {0}/10 positive, {1}/10 negative".format(n_match, n_distinct))
        print(n_match_progress, n_distinct_progress)
        progress = min(1, n_match_progress + n_distinct_progress)
        return progress

    def on_exit(controller, linker=linker):
        linker.train()
        # Save our weights and predicates to disk.  If the settings file
        # exists, we will skip all the training and learning next time we run
        # this file.
        with open(settings_file, 'wb') as sf:
            linker.writeSettings(sf)

        print('clustering...')
        linked_records = linker.match(left_records, right_records, 0)
        print('# duplicate sets', len(linked_records))

        # ## Writing Results

        # Write our original data back out to a CSV with a new column called 
        # 'Cluster ID' which indicates which records refer to each other.

        cluster_membership = {}
        cluster_id = None
        for cluster_id, (cluster, score) in enumerate(linked_records):
            for record_id in cluster:
                cluster_membership[record_id] = (cluster_id, score)

        if cluster_id:
            unique_id = cluster_id + 1
        else:
            unique_id = 0


        with open(output_file, 'w') as f:
            writer = csv.writer(f)

            header_unwritten = True

            for fileno, filename in enumerate((left_record_file_path, right_record_file_path)):
                with open(filename) as f_input:
                    reader = csv.reader(f_input)

                    if header_unwritten:
                        heading_row = next(reader)
                        heading_row.insert(0, 'source file')
                        heading_row.insert(0, 'Link Score')
                        heading_row.insert(0, 'Cluster ID')
                        writer.writerow(heading_row)
                        header_unwritten = False
                    else:
                        next(reader)

                    for row_id, row in enumerate(reader):
                        cluster_details = cluster_membership.get(filename + str(row_id))
                        if cluster_details is None:
                            cluster_id = unique_id
                            unique_id += 1
                            score = None
                        else:
                            cluster_id, score = cluster_details
                        row.insert(0, fileno)
                        row.insert(0, score)
                        row.insert(0, cluster_id)
                        writer.writerow(row)

        print(cluster_membership)

    stream = record_pairs_stream(linker)

    with open('./record_pairs.html') as template_file:
        html_template = template_file.read()

    return {
        'view_id': 'html',
        'dataset': dataset,
        'stream': stream,
        'update': update,
        'progress': get_progress,
        'on_exit': on_exit,
        'config': {
            'html_template': html_template
        }
    }
