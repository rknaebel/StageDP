import os
import sys

from nltk import Tree

from eval.metrics import Metrics
from models.parser import RstParser
from models.tree import RstTree
from utils.document import Doc


class Evaluator:
    def __init__(self, model_dir):
        sys.stderr.write('Load parsing models ...\n')
        self.parser = RstParser.load(model_dir)

    def parse(self, doc):
        """ Parse one document using the given parsing models"""
        pred_rst = self.parser.sr_parse(doc)
        return pred_rst

    @staticmethod
    def writebrackets(fname, brackets):
        """ Write the bracketing results into file"""
        with open(fname, 'w') as fout:
            for item in brackets:
                fout.write(str(item) + '\n')

    def eval_parser(self, path, bcvocab=None):
        """ Test the parsing performance"""
        met = Metrics()
        for fmerge, pred_rst in self.parse_docs(path, bcvocab):
            pred_brackets = pred_rst.bracketing()
            fbrackets = fmerge.replace('.merge', '.brackets')
            # Write brackets into file
            Evaluator.writebrackets(fbrackets, pred_brackets)
            fdis = fmerge.replace('.merge', '.dis')
            gold_rst = RstTree.from_file(fdis, fmerge)
            met.eval(gold_rst, pred_rst)
        met.report()

    def draw_parse_results(self, path, bcvocab=None):
        from nltk.draw.tree import TreeWidget
        from nltk.draw.util import CanvasFrame
        for fmerge, pred_rst in self.parse_docs(path, bcvocab):
            fname = fmerge.replace(".merge", ".ps")
            tree_str = pred_rst.get_parse()
            if not fname.endswith(".ps"):
                fname += ".ps"
            cf = CanvasFrame()
            t = Tree.fromstring(tree_str)
            tc = TreeWidget(cf.canvas(), t)
            tc['node_font'] = 'arial 14 bold'
            tc['leaf_font'] = 'arial 14'
            tc['node_color'] = '#005990'
            tc['leaf_color'] = '#3F8F57'
            tc['line_color'] = '#175252'
            cf.add_widget(tc, 10, 10)  # (10,10) offsets
            cf.print_to_file(fname)
            cf.destroy()
            pprint_tree_str = Tree.fromstring(tree_str).pformat(margin=150)
            with open(fmerge.replace(".merge", ".parse"), 'w') as fout:
                fout.write(pprint_tree_str)

    def parse_docs(self, path, bcvocab=None):
        preds = []
        doclist = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('.merge')]
        for fmerge in doclist:
            doc = Doc.from_file(open(fmerge))
            preds.append((fmerge, self.parser.sr_parse(doc, bcvocab)))
        return preds
