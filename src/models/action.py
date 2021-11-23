import gzip
import logging
import pickle
from operator import itemgetter

from sklearn.svm import LinearSVC

from stagedp.utils.other import reverse_dict
from stagedp.utils.other import vectorize


class ActionClassifier:
    def __init__(self, feature_template, actionxid_map):
        self.feature_template = feature_template
        self.actionxid_map = actionxid_map
        self.idxaction_map = reverse_dict(actionxid_map)
        self.classifier = LinearSVC(C=1.0, penalty='l1', loss='squared_hinge', dual=False, tol=1e-7, max_iter=1000)

    def train(self, feature_matrix, action_labels):
        """ Perform batch-learning on parsing models action classifier
        """
        logging.info('Training classifier for action...')
        self.classifier.fit(feature_matrix, action_labels)

    def predict_probs(self, features):
        """ predict labels and rank the decision label with their confidence
            value, output labels and probabilities
        """
        vec = vectorize(features, self.feature_template)
        vals = self.classifier.decision_function(vec)
        action_vals = {}
        for idx in range(len(self.idxaction_map)):
            action_vals[self.idxaction_map[idx]] = vals[0, idx]
        sorted_actions = sorted(action_vals.items(), key=itemgetter(1), reverse=True)
        return sorted_actions

    def save(self, fname):
        """ Save models
        """
        if not fname.endswith('.gz'):
            fname += '.gz'
        data = {'action_clf': self.classifier,
                'feature_template': self.feature_template,
                'actionxid_map': self.actionxid_map}
        with gzip.open(fname, 'wb') as fout:
            pickle.dump(data, fout)
        logging.info('Save action classifier into file: '
                     '{} with {} features and {} actions.'.format(fname,
                                                                  len(self.feature_template), len(self.actionxid_map)))

    @staticmethod
    def load(fname):
        """ Load models
        """
        data = pickle.load(gzip.open(fname, 'rb'))
        feature_template = data['feature_template']
        actionxid_map = data['actionxid_map']
        clf = ActionClassifier(feature_template, actionxid_map)
        clf.classifier = data['action_clf']
        logging.info('Load action classifier from file: '
                     '{} with {} features and {} actions.'.format(fname, len(feature_template), len(actionxid_map)))
        return clf
