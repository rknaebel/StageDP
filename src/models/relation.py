import gzip
import logging
import pickle

from sklearn.svm import LinearSVC

from stagedp.utils.other import reverse_dict
from stagedp.utils.other import vectorize


class RelationClassifier:
    def __init__(self, feature_template_level_0, feature_template_level_1, feature_template_level_2, relationxid_map):
        self.feature_template_level_0 = feature_template_level_0
        self.feature_template_level_1 = feature_template_level_1
        self.feature_template_level_2 = feature_template_level_2
        self.relationxid_map = relationxid_map
        self.idxrelation_map = reverse_dict(relationxid_map)
        self.classifier_level_0 = LinearSVC(C=1.0, penalty='l1', loss='squared_hinge', dual=False, tol=1e-7,
                                            max_iter=1000)
        self.classifier_level_1 = LinearSVC(C=1.0, penalty='l1', loss='squared_hinge', dual=False, tol=1e-7,
                                            max_iter=1000)
        self.classifier_level_2 = LinearSVC(C=1.0, penalty='l1', loss='squared_hinge', dual=False, tol=1e-7,
                                            max_iter=1000)

    def train(self, feature_matrix, relation_labels, level):
        """ Perform batch-learning on parsing models relation classifier
        """
        logging.info('Training classifier for relation at level {}...'.format(level))
        if level == 0:
            self.classifier_level_0.fit(feature_matrix, relation_labels)
        if level == 1:
            self.classifier_level_1.fit(feature_matrix, relation_labels)
        if level == 2:
            self.classifier_level_2.fit(feature_matrix, relation_labels)

    def predict(self, features, level):
        if level == 0:
            vec = vectorize(features, self.feature_template_level_0)
            pred_label = self.classifier_level_0.predict(vec)
        if level == 1:
            vec = vectorize(features, self.feature_template_level_1)
            pred_label = self.classifier_level_1.predict(vec)
        if level == 2:
            vec = vectorize(features, self.feature_template_level_2)
            pred_label = self.classifier_level_2.predict(vec)
        return self.idxrelation_map[pred_label[0]]

    def save(self, fname):
        """ Save models
        """
        if not fname.endswith('.gz'):
            fname += '.gz'
        data = {'relation_clf_level_0': self.classifier_level_0,
                'relation_clf_level_1': self.classifier_level_1,
                'relation_clf_level_2': self.classifier_level_2,
                'feature_template_level_0': self.feature_template_level_0,
                'feature_template_level_1': self.feature_template_level_1,
                'feature_template_level_2': self.feature_template_level_2,
                'relationxid_map': self.relationxid_map}
        with gzip.open(fname, 'wb') as fout:
            pickle.dump(data, fout)
        logging.info('Save relation classifier into file: {} with {} features at level 0, {} features at level 1, '
                     '{} features at level 2, and {} relations.'.format(fname,
                                                                        len(self.feature_template_level_0),
                                                                        len(self.feature_template_level_1),
                                                                        len(self.feature_template_level_2),
                                                                        len(self.idxrelation_map)))

    @staticmethod
    def load(fname):
        """ Load models
        """
        data = pickle.load(gzip.open(fname, 'rb'))
        classifier_level_0 = data['relation_clf_level_0']
        classifier_level_1 = data['relation_clf_level_1']
        classifier_level_2 = data['relation_clf_level_2']
        feature_template_level_0 = data['feature_template_level_0']
        feature_template_level_1 = data['feature_template_level_1']
        feature_template_level_2 = data['feature_template_level_2']
        relationxid_map = data['relationxid_map']
        clf = RelationClassifier(
            feature_template_level_0, feature_template_level_1, feature_template_level_2, relationxid_map)
        clf.classifier_level_0 = classifier_level_0
        clf.classifier_level_1 = classifier_level_1
        clf.classifier_level_2 = classifier_level_2
        logging.info('Load relation classifier from file: {} with {} features at level 0, {} features at level 1, '
                     '{} features at level 2, and {} relations.'.format(fname,
                                                                        len(feature_template_level_0),
                                                                        len(feature_template_level_1),
                                                                        len(feature_template_level_2),
                                                                        len(relationxid_map)))
        return clf
