import os

from stagedp.features.extraction import ActionFeatureGenerator, RelationFeatureGenerator
from stagedp.models.action import ActionClassifier
from stagedp.models.relation import RelationClassifier
from stagedp.models.state import ParsingState
from stagedp.models.tree import RstTree


class RstParser:
    def __init__(self, action_clf, relation_clf):
        self.action_clf: ActionClassifier = action_clf
        self.relation_clf: RelationClassifier = relation_clf

    def train(self, rst_train, brown_clusters):
        self.action_clf.train(rst_train, brown_clusters)
        self.relation_clf.train(rst_train, brown_clusters)

    def save(self, model_dir):
        """Save models
        """
        self.action_clf.save(os.path.join(model_dir, 'model.action.gz'))
        self.relation_clf.save(os.path.join(model_dir, 'model.relation.gz'))

    @staticmethod
    def load(model_dir):
        """ Load models
        """
        action_clf = ActionClassifier.load(os.path.join(model_dir, 'model.action.gz'))
        relation_clf = RelationClassifier.load(os.path.join(model_dir, 'model.relation.gz'))
        return RstParser(action_clf, relation_clf)

    def sr_parse(self, doc, bcvocab=None):
        """ Shift-reduce RST parsing based on models prediction

        :type doc: Doc
        :param doc: the document instance

        :type bcvocab: dict
        :param bcvocab: brown clusters
        """
        # use transition-based parsing to build tree structure
        conf = ParsingState([], [])
        conf.init(doc)
        action_hist = []
        while not conf.end_parsing():
            stack, queue = conf.get_status()
            action_feats = ActionFeatureGenerator(stack, queue, action_hist, doc, bcvocab).gen_features()
            action_probs = self.action_clf.predict_probs(action_feats)
            for action, cur_prob in action_probs:
                if conf.is_action_allowed(action):
                    conf.operate(action)
                    action_hist.append(action)
                    break
        tree = conf.get_parse_tree()
        # assign the node to rst_tree
        rst_tree = RstTree(tree, doc)
        # tag relations for the tree
        for node in rst_tree.postorder():
            if (node.lnode is not None) and (node.rnode is not None):
                fg = RelationFeatureGenerator(node, rst_tree, node.level, bcvocab)
                relation_feats = fg.gen_features()
                relation = self.relation_clf.predict(relation_feats, node.level)
                node.assign_relation(relation)
        return rst_tree

    @staticmethod
    def from_data(rst_train, brown_clusters):
        action_clf = ActionClassifier.from_data(rst_train, brown_clusters)
        relation_clf = RelationClassifier.from_data(rst_train, brown_clusters)
        return RstParser(action_clf, relation_clf)
