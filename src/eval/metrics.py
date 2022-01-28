import numpy


class Performance:
    def __init__(self):
        self.percision = []
        self.recall = []
        self.hit_num = 0


class Metrics:
    def __init__(self):
        self.span_perf = Performance()
        self.nuc_perf = Performance()
        self.rela_perf = Performance()
        self.span_num = 0
        self.hit_num_each_relation = {}
        self.pred_num_each_relation = {}
        self.gold_num_each_relation = {}

    def eval(self, goldtree, predtree):
        """ Evaluation performance on one pair of RST trees

        :type goldtree: RSTTree class
        :param goldtree: gold RST tree

        :type predtree: RSTTree class
        :param predtree: RST tree from the parsing algorithm
        """
        goldbrackets = goldtree.bracketing()
        predbrackets = predtree.bracketing()
        self.span_num += len(goldbrackets)
        self._eval(goldbrackets, predbrackets, idx=1)
        self._eval(goldbrackets, predbrackets, idx=2)
        self._eval(goldbrackets, predbrackets, idx=3)

    def _eval(self, goldbrackets, predbrackets, idx):
        """ Evaluation on each discourse span
        """
        # goldspan = [item[:idx] for item in goldbrackets]
        # predspan = [item[:idx] for item in predbrackets]
        if idx == 1 or idx == 2:
            goldspan = [item[:idx] for item in goldbrackets]
            predspan = [item[:idx] for item in predbrackets]
        elif idx == 3:
            goldspan = [(item[0], item[2]) for item in goldbrackets]
            predspan = [(item[0], item[2]) for item in predbrackets]
        else:
            raise ValueError('Undefined idx for evaluation')
        hitspan = [span for span in goldspan if span in predspan]
        p, r = 0.0, 0.0
        for span in hitspan:
            if span in goldspan:
                p += 1.0
            if span in predspan:
                r += 1.0
        if idx == 1:
            self.span_perf.hit_num += p
        elif idx == 2:
            self.nuc_perf.hit_num += p
        elif idx == 3:
            self.rela_perf.hit_num += p
        p /= len(goldspan)
        r /= len(predspan)
        if idx == 1:
            self.span_perf.percision.append(p)
            self.span_perf.recall.append(r)
        elif idx == 2:
            self.nuc_perf.percision.append(p)
            self.nuc_perf.recall.append(r)
        elif idx == 3:
            self.rela_perf.percision.append(p)
            self.rela_perf.recall.append(r)
        if idx == 3:
            for span in hitspan:
                relation = span[-1]
                if relation in self.hit_num_each_relation:
                    self.hit_num_each_relation[relation] += 1
                else:
                    self.hit_num_each_relation[relation] = 1
            for span in goldspan:
                relation = span[-1]
                if relation in self.gold_num_each_relation:
                    self.gold_num_each_relation[relation] += 1
                else:
                    self.gold_num_each_relation[relation] = 1
            for span in predspan:
                relation = span[-1]
                if relation in self.pred_num_each_relation:
                    self.pred_num_each_relation[relation] += 1
                else:
                    self.pred_num_each_relation[relation] = 1

    def report_part(self, part, part_label):
        p = numpy.array(part.percision).mean()
        # r = numpy.array(part.recall).mean()
        # f1 = (2 * p * r) / (p + r)
        print(f'Average precision on {part_label} level is {p:.4f}')
        # print('Recall on span level is {0:.4f}'.format(r))
        # print('F1 score on span level is {0:.4f}'.format(f1))
        print(f'Global precision on {part_label} level is {part.hit_num / self.span_num:.4f}')

    def report(self):
        """ Compute the F1 score for different eval levels
            and print it out
        """
        self.report_part(self.span_perf, "span")
        self.report_part(self.nuc_perf, "nuclearity")
        self.report_part(self.rela_perf, "relation")
        # sorted_relations = sorted(self.gold_num_each_relation.keys(), key=lambda x: self.gold_num_each_relation[x])
        sorted_relations = sorted(self.gold_num_each_relation.keys())
        print("= " * 55)
        for relation in sorted_relations:
            hit_num = self.hit_num_each_relation[relation] if relation in self.hit_num_each_relation else 0
            gold_num = self.gold_num_each_relation[relation]
            pred_num = self.pred_num_each_relation[relation] if relation in self.pred_num_each_relation else 0
            precision = hit_num / pred_num if pred_num > 0 else 0
            recall = hit_num / gold_num
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            print(f'Relation\t{relation:20}\tgold_num\t{gold_num:4d}\t'
                  f'precision\t{precision:05.4f}\trecall\t{recall:05.4f}\tf1\t{f1:05.4f}')
