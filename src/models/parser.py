#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 10/27/2016 下午7:32
import os

from stagedp.features.extraction import ActionFeatureGenerator, RelationFeatureGenerator
from stagedp.models import ActionClassifier, RelationClassifier
from stagedp.models.state import ParsingState
from stagedp.models.tree import RstTree


class RstParser:
    def __init__(self, action_clf, relation_clf):
        self.action_clf = action_clf
        self.relation_clf = relation_clf

    def save(self, model_dir):
        """Save models
        """
        self.action_clf.save(os.path.join(model_dir, 'model.action.gz'))
        self.relation_clf.save(os.path.join(model_dir, 'model.relation.gz'))

    def load(self, model_dir):
        """ Load models
        """
        self.action_clf.load(os.path.join(model_dir, 'model.action.gz'))
        self.relation_clf.load(os.path.join(model_dir, 'model.relation.gz'))

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
            fg = ActionFeatureGenerator(stack, queue, action_hist, doc, bcvocab)
            action_feats = fg.gen_features()
            action_probs = self.action_clf.predict_probs(action_feats)
            for action, cur_prob in action_probs:
                if conf.is_action_allowed(action):
                    conf.operate(action)
                    action_hist.append(action)
                    break
        tree = conf.get_parse_tree()
        # RstTree.down_prop(tree)
        # assign the node to rst_tree
        rst_tree = RstTree()
        rst_tree.assign_tree(tree)
        rst_tree.assign_doc(doc)
        rst_tree.down_prop(tree)
        rst_tree.back_prop(tree, doc)
        # tag relations for the tree
        post_nodelist = RstTree.postorder_DFT(rst_tree.tree, [])
        for node in post_nodelist:
            if (node.lnode is not None) and (node.rnode is not None):
                fg = RelationFeatureGenerator(node, rst_tree, node.level, bcvocab)
                relation_feats = fg.gen_features()
                relation = self.relation_clf.predict(relation_feats, node.level)
                node.assign_relation(relation)
        return rst_tree
