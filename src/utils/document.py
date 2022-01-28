from collections import defaultdict

from utils.token import Token


class Doc:
    """ Build one doc instance from *.merge file
    """

    def __init__(self):
        self.token_dict = None
        self.edu_dict = None

    @staticmethod
    def from_file(fmerge):
        """ Read information from the merge file, and create an Doc instance
        """
        doc = Doc()
        doc.token_dict = {}
        for line in fmerge:
            line = line.strip()
            if len(line) == 0:
                continue
            tok = doc._parse_fmerge_line(line)
            doc.token_dict[len(doc.token_dict)] = tok
        # Get EDUs from tokendict
        doc.edu_dict = doc._recover_edus(doc.token_dict)
        return doc

    def init_from_tokens(self, token_list):
        self.token_dict = {idx: token for idx, token in enumerate(token_list)}
        self.edu_dict = self._recover_edus(self.token_dict)

    @staticmethod
    def _parse_fmerge_line(line):
        """ Parse one line from *.merge file
        """
        sent_i, tok_i, text, lemma, upos, xpos, deprel, head, _, _, edu_i, par_i = line.split("\t")
        tok = Token()
        tok.pidx, tok.sidx, tok.tidx = int(par_i), int(sent_i), int(tok_i)
        tok.word, tok.lemma = text, lemma
        tok.pos = xpos
        tok.dep_label = deprel
        tok.hidx = int(head)
        # tok.ner, tok.partial_parse = items[7], items[8]
        tok.eduidx = int(edu_i)
        return tok

    @staticmethod
    def _recover_edus(token_dict):
        """ Recover EDUs from token_dict
        """
        edu_dict = defaultdict(list)
        for gidx, token in token_dict.items():
            edu_dict[token.eduidx].append(gidx)
        return dict(edu_dict)
