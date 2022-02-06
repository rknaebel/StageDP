import gzip
import io
import logging
import pickle

import click
from nltk import Tree

from stagedp.models.parser import RstParser
from stagedp.utils.annotation import load_parser, merge_edus_into_parses, merge_as_text
from stagedp.utils.document import Doc


@click.command()
@click.argument('edu_file', type=str)
@click.argument('model_path', type=str)
@click.option('-o', '--output', default='-', type=click.File('w'))
@click.option('--brown_clusters', default="../data/resources/bc3200.pickle.gz", help='brown cluster file')
def main(edu_file, model_path, output, brown_clusters):
    logging.basicConfig(level=logging.INFO)
    rst_parser = RstParser.load(model_path)
    with gzip.open(brown_clusters) as fin:
        logging.info('Load Brown clusters for creating features ...')
        brown_clusters = pickle.load(fin)
    parser = load_parser()
    edus = [edu.strip() for edu in open(edu_file)]
    text = ' '.join(edus).replace('<P>', '')
    parses = parser(text)
    parses = merge_as_text(merge_edus_into_parses(edus, parses))
    doc = Doc.from_file(io.StringIO(parses))
    pred_rst = rst_parser.sr_parse(doc, brown_clusters)
    tree_str = pred_rst.get_parse()
    pprint_tree_str = Tree.fromstring(tree_str).pformat(margin=180)
    output.write(pprint_tree_str + "\n")


if __name__ == '__main__':
    main()
