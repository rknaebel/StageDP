import os
import sys

from stagedp.utils.document import Doc
from stagedp.utils.span import SpanNode


class RstTree:
    def __init__(self, tree, doc):
        self.binary = True
        self.tree: SpanNode = tree
        self.doc = doc
        self.down_prop(self.tree)
        self.back_prop(self.tree, self.doc)

    @staticmethod
    def from_file(fdis, fmerge):
        """ Build BINARY RST tree
        """
        with open(fdis) as fin:
            text = fin.read()
        tree = RstTree.binarize_tree(RstTree.build_tree(text))
        doc = Doc.from_file(open(fmerge))
        return RstTree(tree, doc)

    @staticmethod
    def read_rst_trees(data_dir):
        files = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.dis')]
        rst_trees = []
        for fdis in files:
            fmerge = fdis.replace('.dis', '.merge')
            if not os.path.isfile(fmerge):
                raise FileNotFoundError('Corresponding .fmerge file does not exist. You should do preprocessing first.')
            rst_trees.append(RstTree.from_file(fdis, fmerge))
        return rst_trees

    def convert_node_to_str(self, node, sep=' '):
        text = node.text
        words = [self.doc.token_dict[tidx].word for tidx in text]
        return sep.join(words)

    def get_edu_node(self):
        """ Get all left nodes. It can be used for generating training
            examples from gold RST tree

        :type tree: SpanNode instance
        :param tree: an binary RST tree
        """
        edulist = []
        for node in self.postorder():
            if (node.lnode is None) and (node.rnode is None):
                edulist.append(node)
        return edulist

    @staticmethod
    def build_tree(text):
        """ Build tree from *.dis file

        :type text: string
        :param text: RST tree read from a *.dis file
        """
        tokens = text.strip().replace('//TT_ERR', '').replace('\n', '').replace('(', ' ( ').replace(')', ' ) ').split()
        queue = RstTree.process_text(tokens)
        stack = []
        while queue:
            token = queue.pop(0)
            if token == ')':
                # If ')', start processing
                content = []  # Content in the stack
                while stack:
                    cont = stack.pop()
                    if cont == '(':
                        break
                    else:
                        content.append(cont)
                content.reverse()  # Reverse to the original order
                # Parse according to the first content word
                if len(content) < 2:
                    raise ValueError("content = {}".format(content))
                label = content.pop(0)
                if label == 'Root':
                    node = SpanNode(prop=label)
                    node.create_node(content)
                    stack.append(node)
                elif label == 'Nucleus':
                    node = SpanNode(prop=label)
                    node.create_node(content)
                    stack.append(node)
                elif label == 'Satellite':
                    node = SpanNode(prop=label)
                    node.create_node(content)
                    stack.append(node)
                elif label == 'span':
                    # Merge
                    beginindex = int(content.pop(0))
                    endindex = int(content.pop(0))
                    stack.append(('span', beginindex, endindex))
                elif label == 'leaf':
                    # Merge
                    eduindex = int(content.pop(0))
                    RstTree.check_content(label, content)
                    stack.append(('leaf', eduindex, eduindex))
                elif label == 'rel2par':
                    # Merge
                    relation = content.pop(0)
                    RstTree.check_content(label, content)
                    stack.append(('relation', relation))
                elif label == 'text':
                    # Merge
                    txt = RstTree.create_text(content)
                    stack.append(('text', txt))
                else:
                    raise ValueError(
                        "Unrecognized parsing label: {} \n\twith content = {}\n\tstack={}\n\tqueue={}".format(label,
                                                                                                              content,
                                                                                                              stack,
                                                                                                              queue))
            else:
                # else, keep push into the stack
                stack.append(token)
        return stack[-1]

    @staticmethod
    def process_text(tokens):
        """ Preprocessing token list for filtering '(' and ')' in text
        :type tokens: list
        :param tokens: list of tokens
        """
        identifier = '_!'
        within_text = False
        for (idx, tok) in enumerate(tokens):
            if identifier in tok:
                for _ in range(tok.count(identifier)):
                    within_text = not within_text
            if ('(' in tok) and within_text:
                tok = tok.replace('(', '-LB-')
            if (')' in tok) and within_text:
                tok = tok.replace(')', '-RB-')
            tokens[idx] = tok
        return tokens

    @staticmethod
    def create_text(lst):
        """ Create text from a list of tokens

        :type lst: list
        :param lst: list of tokens
        """
        newlst = []
        for item in lst:
            item = item.replace("_!", "")
            newlst.append(item)
        text = ' '.join(newlst)
        # Lower-casing
        return text.lower()

    @staticmethod
    def check_content(label, c):
        """ Check whether the content is legal

        :type label: string
        :param label: parsing label, such 'span', 'leaf'

        :type c: list
        :param c: list of tokens
        """
        if len(c) > 0:
            raise ValueError("{} with content={}".format(label, c))

    @staticmethod
    def binarize_tree(tree):
        """ Convert a general RST tree to a binary RST tree

        :type tree: instance of SpanNode
        :param tree: a general RST tree
        """
        queue = [tree]
        while queue:
            node = queue.pop(0)
            queue += node.nodelist
            # Construct binary tree
            if len(node.nodelist) == 2:
                node.lnode = node.nodelist[0]
                node.rnode = node.nodelist[1]
                # Parent node
                node.lnode.pnode = node
                node.rnode.pnode = node
            elif len(node.nodelist) > 2:
                # Remove one node from the nodelist
                node.lnode = node.nodelist.pop(0)
                newnode = SpanNode(node.nodelist[0].prop)
                newnode.nodelist += node.nodelist
                # Right-branching
                node.rnode = newnode
                # Parent node
                node.lnode.pnode = node
                node.rnode.pnode = node
                # Add to the head of the queue
                # So the code will keep branching
                # until the nodelist size is 2
                queue.insert(0, newnode)
            # Clear nodelist for the current node
            node.nodelist = []
        return tree

    @staticmethod
    def back_prop(tree, doc):
        """ Starting from leaf node, propagating node
            information back to root node

        :type tree: SpanNode instance
        :param tree: an binary RST tree
        """
        tree_nodes = RstTree.BFTbin(tree)
        tree_nodes.reverse()
        for node in tree_nodes:
            if (node.lnode is not None) and (node.rnode is not None):
                # Non-leaf node
                node.edu_span = RstTree.__getspaninfo(node.lnode, node.rnode)
                node.text = RstTree.__gettextinfo(doc.edu_dict, node.edu_span)
                if node.relation is None:
                    # If it is a new node created by binarization
                    if node.prop == 'Root':
                        pass
                    else:
                        node.relation = RstTree.__getrelationinfo(node.lnode, node.rnode)
                node.form, node.nuc_span, node.nuc_edu = RstTree.__getforminfo(node.lnode, node.rnode)
                node.height = max(node.lnode.height, node.rnode.height) + 1
                node.max_depth = max(node.lnode.max_depth, node.rnode.max_depth)
                if node.form == 'NS':
                    node.child_relation = node.rnode.relation
                else:
                    node.child_relation = node.lnode.relation
                if doc.token_dict[node.lnode.text[0]].sidx == doc.token_dict[node.rnode.text[-1]].sidx:
                    node.level = 0
                elif doc.token_dict[node.lnode.text[0]].pidx == doc.token_dict[node.rnode.text[-1]].pidx:
                    node.level = 1
                else:
                    node.level = 2
            elif (node.lnode is None) and (node.rnode is not None):
                raise ValueError("Unexpected left node")
            elif (node.lnode is not None) and (node.rnode is None):
                raise ValueError("Unexpected right node")
            else:
                # Leaf node
                node.text = RstTree.__gettextinfo(doc.edu_dict, node.edu_span)
                node.height = 0
                node.max_depth = node.depth
                node.level = 0

    @staticmethod
    def down_prop(tree):
        """
        Starting from root node, propagating node information down to leaf nodes
        :param tree: SpanNode instance
        :param doc: Doc instance
        :return: root node
        """
        tree_nodes = RstTree.BFTbin(tree)
        root_node = tree_nodes.pop(0)
        root_node.depth = 0
        for node in tree_nodes:
            assert node.pnode.depth >= 0
            node.depth = node.pnode.depth + 1

    @staticmethod
    def BFTbin(tree):
        """ Breadth-first treavsal on binary RST tree

        :type tree: SpanNode instance
        :param tree: an binary RST tree
        """
        queue = [tree]
        bft_nodelist = []
        while queue:
            node = queue.pop(0)
            bft_nodelist.append(node)
            if node.lnode is not None:
                queue.append(node.lnode)
            if node.rnode is not None:
                queue.append(node.rnode)
        return bft_nodelist

    def postorder(self):
        """ Post order traversal on binary RST tree

        :type tree: SpanNode instance
        :param tree: an binary RST tree

        """

        def _postorder(tree, nodelist):
            if tree.lnode is not None:
                _postorder(tree.lnode, nodelist)
            if tree.rnode is not None:
                _postorder(tree.rnode, nodelist)
            nodelist.append(tree)

        node_list = []
        _postorder(self.tree, node_list)
        return node_list

    @staticmethod
    def __getspaninfo(lnode, rnode):
        """ Get span size for parent node

        :type lnode,rnode: SpanNode instance
        :param lnode,rnode: Left/Right children nodes
        """
        try:
            edu_span = (lnode.edu_span[0], rnode.edu_span[1])
            return edu_span
        except TypeError:
            print(lnode.prop, rnode.prop)
            print(lnode.nuc_span, rnode.nuc_span)
            sys.exit()

    @staticmethod
    def __getforminfo(lnode, rnode):
        """ Get Nucleus/Satellite form and Nucleus span

        :type lnode,rnode: SpanNode instance
        :param lnode,rnode: Left/Right children nodes
        """
        if (lnode.prop == 'Nucleus') and (rnode.prop == 'Satellite'):
            nuc_span = lnode.edu_span
            nuc_edu = lnode.nuc_edu
            form = 'NS'
        elif (lnode.prop == 'Satellite') and (rnode.prop == 'Nucleus'):
            nuc_span = rnode.edu_span
            nuc_edu = rnode.nuc_edu
            form = 'SN'
        elif (lnode.prop == 'Nucleus') and (rnode.prop == 'Nucleus'):
            nuc_span = (lnode.edu_span[0], rnode.edu_span[1])
            nuc_edu = lnode.nuc_edu
            form = 'NN'
        else:
            raise ValueError("")
        return form, nuc_span, nuc_edu

    @staticmethod
    def __getrelationinfo(lnode, rnode):
        """ Get relation information

        :type lnode,rnode: SpanNode instance
        :param lnode,rnode: Left/Right children nodes
        """
        if (lnode.prop == 'Nucleus') and (rnode.prop == 'Nucleus'):
            relation = lnode.relation
        elif (lnode.prop == 'Nucleus') and (rnode.prop == 'Satellite'):
            relation = lnode.relation
        elif (lnode.prop == 'Satellite') and (rnode.prop == 'Nucleus'):
            relation = rnode.relation
        else:
            print('lnode.prop = {}, lnode.edu_span = {}'.format(lnode.prop, lnode.edu_span))
            print('rnode.prop = {}, lnode.edu_span = {}'.format(rnode.prop, rnode.edu_span))
            raise ValueError("Error when find relation for new node")
        return relation

    @staticmethod
    def __gettextinfo(edu_dict, edu_span):
        """ Get text span for parent node

        :type edu_dict: dict of list
        :param edu_dict: EDU from this document

        :type edu_span: tuple with two elements
        :param edu_span: start/end of EDU IN this span
        """
        # text = lnode.text + " " + rnode.text
        text = []
        for idx in range(edu_span[0], edu_span[1] + 1):
            text += edu_dict[idx]
        # Return: A list of token indices
        return text

    def get_parse(self):
        """ Get parse tree in dis format.
        """
        type_map = {n[0]: n for n in ['Nucleus', 'Satellite']}

        def get_form(form):
            return type_map[form[0]], type_map[form[1]]

        def _helper(node, node_form):
            if node.is_leaf():
                return f"({node_form} (leaf {node.edu_span[0]}) (rel2par {node.relation}) (text _!{self.convert_node_to_str(node, sep='_')}_!))"
            else:
                lnode_form, rnode_form = get_form(node.form)
                if node_form != 'Root':
                    return f"({node_form} (span {node.edu_span}) (rel2par {node.relation}) {_helper(node.lnode, lnode_form)} {_helper(node.rnode, rnode_form)})"
                else:
                    return f"(Root (span {node.edu_span}) {_helper(node.lnode, lnode_form)} {_helper(node.rnode, rnode_form)})"

        return _helper(self.tree, 'Root')

    def bracketing(self):
        """ Generate brackets according a Binary RST tree
        """
        nodelist = self.postorder()
        nodelist.pop()  # Remove the root node
        brackets = []
        for node in nodelist:
            relation = node.relation
            b = (node.edu_span, node.prop, relation)
            brackets.append(b)
        return brackets
