import gzip
import logging
import pickle
from collections import Counter

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from stagedp.features.extraction import RelationFeatureGenerator
from stagedp.utils.other import reverse_dict


class RelationClassifier:
    def __init__(self, relationxid_map):
        self.relationxid_map = relationxid_map
        self.idxrelation_map = reverse_dict(relationxid_map)
        self.models = [
            Pipeline([
                ('vectorizer', DictVectorizer()),
                # ('variance', VarianceThreshold(threshold=0.0001)),
                ('model', SGDClassifier(loss='log', penalty='l2', average=32, tol=1e-7, max_iter=1000, n_jobs=-1,
                                        class_weight='balanced'))
            ]),
            Pipeline([
                ('vectorizer', DictVectorizer()),
                # ('variance', VarianceThreshold(threshold=0.0001)),
                ('model', SGDClassifier(loss='log', penalty='l2', average=32, tol=1e-7, max_iter=1000, n_jobs=-1,
                                        class_weight='balanced'))
            ]),
            Pipeline([
                ('vectorizer', DictVectorizer()),
                # ('variance', VarianceThreshold(threshold=0.0001)),
                ('model', SGDClassifier(loss='log', penalty='l2', average=32, tol=1e-7, max_iter=1000, n_jobs=-1,
                                        class_weight='balanced'))
            ])
        ]

    def train(self, rst_tree_instances, brown_clusters):
        """ Perform batch-learning on parsing models relation classifier
        """
        for level in [0, 1, 2]:
            logging.info('Training classifier for relation at level {}...'.format(level))
            relation_fvs, relation_labels = list(zip(*self.gen_train_data(rst_tree_instances,
                                                                          brown_clusters, level)))
            logging.info('{} relation samples at level {}.'.format(len(relation_labels), level))
            self.models[level].fit(relation_fvs, relation_labels)

    def predict(self, features, level):
        pred_label = self.models[level].predict([features])[0]
        return self.idxrelation_map[pred_label]

    def save(self, fname):
        """ Save models
        """
        if not fname.endswith('.gz'):
            fname += '.gz'
        data = {'models': self.models,
                'relationxid_map': self.relationxid_map}
        with gzip.open(fname, 'wb') as fout:
            pickle.dump(data, fout)
        logging.info('Save relation classifier into file: {} with {} features at level 0, {} features at level 1, '
                     '{} features at level 2, and {} relations.'.format(fname,
                                                                        self.models[0]['model'].n_features_in_,
                                                                        self.models[1]['model'].n_features_in_,
                                                                        self.models[2]['model'].n_features_in_,
                                                                        len(self.idxrelation_map)))

    @staticmethod
    def load(fname):
        """ Load models
        """
        data = pickle.load(gzip.open(fname, 'rb'))
        models = data['models']
        relationxid_map = data['relationxid_map']
        clf = RelationClassifier(relationxid_map)
        clf.models = models
        logging.info('Load relation classifier from file: {} with {} features at level 0, {} features at level 1, '
                     '{} features at level 2, and {} relations.'.format(fname,
                                                                        models[0]['model'].n_features_in_,
                                                                        models[1]['model'].n_features_in_,
                                                                        models[2]['model'].n_features_in_,
                                                                        len(relationxid_map)))
        return clf

    @staticmethod
    def from_data(rst_tree_instances, brown_clusters):
        relation_cnt = Counter(relation for lvl in [0, 1, 2]
                               for rst_tree in rst_tree_instances
                               for features, relation in generate_relation_samples(rst_tree, brown_clusters, level=lvl))
        relation_map = {a: i for i, a in enumerate(relation_cnt)}
        logging.info('{} types of relations: {}'.format(len(relation_map), relation_map.keys()))
        for relation, cnt in relation_cnt.items():
            logging.info('{}\t{}'.format(relation, cnt))
        return RelationClassifier(relation_map)

    def gen_train_data(self, rst_tree_instances, brown_clusters, level):
        for rst_tree in rst_tree_instances:
            for feats, relation in generate_relation_samples(rst_tree, brown_clusters, level):
                yield feats, self.relationxid_map[relation]


def generate_relation_samples(rst_tree, bcvocab, level):
    """ Generate relation samples from an binary RST tree
    :type bcvocab: dict
    :param bcvocab: brown clusters of words
    """
    for node in rst_tree.postorder():
        if node.level == level and (node.lnode is not None) and (node.rnode is not None):
            relation_feats = RelationFeatureGenerator(node, rst_tree, node.level, bcvocab).gen_features()
            if (node.form == 'NN') or (node.form == 'NS'):
                relation = node.rnode.relation
            else:
                relation = node.lnode.relation
            yield relation_feats, relation
