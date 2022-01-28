import sys
from typing import List

import stanza


def load_parser():
    tmp_stdout = sys.stdout
    sys.stdout = sys.stderr
    stanza.download(lang='en')
    parser = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse')
    sys.stdout = tmp_stdout
    return parser


def merge_edus_into_parses(edus: List[str], parses):
    result = []
    edu_i = 0
    edu = edus[edu_i].replace('<P>', '')
    edu_offset = 0
    edu_length = len(edu)
    par_i = 1
    for sent_i, sent in enumerate(parses.sentences):
        for tok_i, tok in enumerate(sent.words):
            if tok.end_char - edu_offset > edu_length:
                edu_i += 1
                edu = edus[edu_i].replace('<P>', '')
                if edus[edu_i - 1].endswith('<P>'):
                    par_i += 1
                edu_offset = tok.start_char
                edu_length = len(edu)
            result.append((sent_i, tok_i + 1, tok.text, tok.lemma, tok.upos, tok.xpos, tok.deprel, tok.head, '_', '_',
                           edu_i + 1, par_i))
    return result


def merge_as_text(tokens):
    return '\n'.join('\t'.join(map(str, tok)) for tok in tokens) + '\n'
