import sys
from typing import List

import stanza
from conllu import TokenList
from conllu.models import Token, Metadata


def load_parser():
    tmp_stdout = sys.stdout
    sys.stdout = sys.stderr
    stanza.download(lang='en')
    parser = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse,constituency', tokenize_no_ssplit=True)
    sys.stdout = tmp_stdout
    return parser


def merge_edus_into_parses(edus: List[str], parses):
    result = []
    edu_i = 0
    edu = edus[edu_i].replace('<P>', '')
    edu_offset = 0
    edu_length = len(edu)
    par_i = 1
    doc_id = str(hash(' '.join(edus)))
    meta = {
        'newdoc id': doc_id,
    }
    for sent_i, sent in enumerate(parses):
        tokens = []
        for tok_i, tok in enumerate(sent.words):
            misc = {}
            if sent_i == 0 and tok_i == 0:
                misc['BeginSeg'] = 'YES'
            if tok.end_char - edu_offset > edu_length:
                misc['BeginSeg'] = 'YES'
                edu_i += 1
                edu = edus[edu_i].replace('<P>', '')
                if edus[edu_i - 1].endswith('<P>'):
                    meta['newpar id'] = f'{doc_id}-p{par_i}'
                    par_i += 1
                edu_offset = tok.start_char
                edu_length = len(edu)
            tokens.append(Token(
                id=tok_i + 1,
                form=tok.text,
                lemma=tok.lemma,
                upos=tok.upos,
                xpos=tok.xpos,
                head=tok.head,
                deprel=tok.deprel,
                deps='_',
                misc=misc))
        meta['sent_id'] = str(sent_i)
        meta['text'] = sent.text
        meta['parse'] = str(sent.constituency)
        result.append(TokenList(tokens, metadata=Metadata(meta)))
        meta = {}
    return result


def merge_as_text(tokens):
    return '\n'.join('\t'.join(map(str, tok)) for tok in tokens) + '\n'
