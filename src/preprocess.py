import glob
import os
import re
import sys
from typing import List

import click
import nltk
from tqdm import tqdm

from stagedp.utils.other import rel2class


def load_parser():
    import stanza
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


@click.command()
@click.argument('source-path', type=str)
@click.argument('target-path', type=str)
@click.option('-r', '--replace-exist', is_flag=True)
@click.option('-d', '--delete-target', is_flag=True)
def main(source_path: str, target_path: str, replace_exist: bool, delete_target: bool):
    if delete_target:
        for fn in os.listdir(target_path):
            os.remove(os.path.join(target_path, fn))
    os.makedirs(target_path, exist_ok=True)
    parser = load_parser()
    dis_files = glob.glob(f'{source_path}/*.dis')
    for dis_file in tqdm(dis_files):
        dest = os.path.join(target_path, os.path.basename(f"{dis_file[:-len('.dis')]}"))
        if not replace_exist and os.path.exists(f"{dest}.dis"):
            continue
        dis_text = open(dis_file, 'r').read()
        dis_tree = nltk.tree.Tree.fromstring(
            re.sub(r'\s+', ' ', dis_text).replace('//TT_ERR', '').strip(),
            leaf_pattern=r"\_!.+?\_!|[^ ()]+")
        # convert (simplify) relation labels
        for tt in dis_tree.subtrees(filter=lambda t: t.label() == 'rel2par'):
            label = tt.pop()
            tt.insert(0, rel2class.get(label.lower(), label.upper()))
        edus = [edu[2:-2].strip() for edu in dis_tree.leaves() if edu.startswith('_!')]
        text = ' '.join(edus).replace('<P>', '')
        parses = parser(text)
        parses = merge_edus_into_parses(edus, parses)
        with open(f"{dest}.dis", 'w') as fh:
            fh.write(dis_tree.pformat(margin=150).replace('<P>', ''))
        with open(f"{dest}.merge", 'w') as fh:
            for tok in parses:
                fh.write('\t'.join(map(str, tok)) + '\n')


if __name__ == '__main__':
    main()
