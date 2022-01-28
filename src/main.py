import gzip
import logging
import pickle

import click

from stagedp.eval.evaluation import Evaluator
from stagedp.models.parser import RstParser
from stagedp.models.tree import RstTree


@click.command()
@click.option('--train_dir', default='', help='train data directory')
@click.option('--test_dir', default='', help='test data directory')
@click.option('--model_dir', help='model directory')
@click.option('--brown_clusters', default="../data/resources/bc3200.pickle.gz", help='brown cluster file')
def main(train_dir, test_dir, model_dir, brown_clusters):
    logging.basicConfig(level=logging.INFO)
    with gzip.open(brown_clusters) as fin:
        logging.info('Load Brown clusters for creating features ...')
        brown_clusters = pickle.load(fin)
    if train_dir:
        rst_train = RstTree.read_rst_trees(data_dir=train_dir)
        rst_parser = RstParser.from_data(rst_train, brown_clusters)
        rst_parser.train(rst_train, brown_clusters)
        rst_parser.save(model_dir=model_dir)
    if test_dir:
        evaluator = Evaluator(model_dir=model_dir)
        evaluator.eval_parser(path=test_dir, bcvocab=brown_clusters)


if __name__ == '__main__':
    main()
