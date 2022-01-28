import gzip
import logging
import pickle
from collections import Counter
from operator import itemgetter

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from stagedp.features.extraction import ActionFeatureGenerator
from stagedp.models.state import ParsingState
from stagedp.utils.other import reverse_dict


class ActionClassifier:
    def __init__(self, actionxid_map):
        self.actionxid_map = actionxid_map
        self.idxaction_map = reverse_dict(actionxid_map)
        self.model = Pipeline([
            ('vectorizer', DictVectorizer()),
            # ('variance', VarianceThreshold(threshold=0.0001)),
            ('model', SGDClassifier(loss='log', penalty='l2', average=32, tol=1e-7, max_iter=1000, n_jobs=-1,
                                    class_weight='balanced'))
            # ('model', RandomForestClassifier(n_estimators=1000, max_depth=25, min_samples_split=5, min_samples_leaf=3,
            #                                  random_state=0, n_jobs=-1))
        ])

    def train(self, rst_tree_instances, brown_clusters):
        """ Perform batch-learning on parsing models action classifier
        """
        logging.info('Training classifier for action...')
        action_fvs, action_labels = list(zip(*self.generate_train_data(rst_tree_instances, brown_clusters)))
        self.model.fit(action_fvs, action_labels)
        print(self.model.score(action_fvs, action_labels))
        action_preds = self.model.predict(action_fvs)
        print(classification_report(action_labels, action_preds))

    def predict_probs(self, features):
        """ predict labels and rank the decision label with their confidence
            value, output labels and probabilities
        """
        vals = self.model.predict_proba([features])[0]
        action_vals = {}
        for idx in range(len(self.idxaction_map)):
            action_vals[self.idxaction_map[idx]] = vals[idx]
        sorted_actions = sorted(action_vals.items(), key=itemgetter(1), reverse=True)
        return sorted_actions

    def save(self, fname):
        """ Save models
        """
        if not fname.endswith('.gz'):
            fname += '.gz'
        data = {'action_clf': self.model,
                'actionxid_map': self.actionxid_map}
        with gzip.open(fname, 'wb') as fout:
            pickle.dump(data, fout)
        logging.info('Save action classifier into file: '
                     '{} with {} features and {} actions.'.format(fname, self.model['model'].n_features_in_,
                                                                  len(self.actionxid_map)))

    @staticmethod
    def load(fname):
        """ Load models
        """
        data = pickle.load(gzip.open(fname, 'rb'))
        actionxid_map = data['actionxid_map']
        clf = ActionClassifier(actionxid_map)
        clf.model = data['action_clf']
        logging.info('Load action classifier from file: '
                     '{} with {} features and {} actions.'.format(fname, clf.model['model'].n_features_in_,
                                                                  len(actionxid_map)))
        return clf

    @staticmethod
    def from_data(rst_tree_instances, brown_clusters):
        action_cnt = Counter(action for rst_tree in rst_tree_instances
                             for features, action in generate_action_samples(rst_tree, brown_clusters))
        action_map = {a: i for i, a in enumerate(action_cnt)}
        logging.info('{} types of actions: {}'.format(len(action_map), action_map.keys()))
        for action, cnt in action_cnt.items():
            logging.info('{}\t{}'.format(action, cnt))
        return ActionClassifier(action_map)

    def generate_train_data(self, rst_tree_instances, brown_clusters):
        for rst_tree in rst_tree_instances:
            for feats, action in generate_action_samples(rst_tree, brown_clusters):
                yield feats, self.actionxid_map[action]


def generate_action_samples(rst_tree, bcvocab):
    """ Generate action samples from an binary RST tree
    :type bcvocab: dict
    :param bcvocab: brown clusters of words
    """
    # post_nodelist = RstTree.postorder_DFT(rst_tree.tree, [])
    # action_list = []
    # relation_list = []
    action_hist = []
    # Initialize queue and stack
    queue = rst_tree.get_edu_node()
    stack = []
    # Start simulating the shift-reduce parsing
    sr_parser = ParsingState(stack, queue)
    for node in rst_tree.postorder():
        if (node.lnode is None) and (node.rnode is None):
            action = ('Shift', None)
        elif (node.lnode is not None) and (node.rnode is not None):
            form = node.form
            action = ('Reduce', form)
        else:
            raise ValueError("Can not decode Shift-Reduce action")
        stack, queue = sr_parser.get_status()
        # Generate features
        action_feats = ActionFeatureGenerator(stack, queue, action_hist, rst_tree.doc, bcvocab).gen_features()
        yield action_feats, action
        # Change status of stack/queue
        # action and relation are necessary here to avoid change rst_trees
        sr_parser.operate(action)
        action_hist.append(action)
