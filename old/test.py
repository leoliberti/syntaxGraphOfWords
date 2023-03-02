#!/usr/bin/env python3.10

########### IMPORTS ###########
import sys
import os
import math
import json
import gzip
import spacy
import benepar
import pickle
import pprint
import ast
import subprocess
import graphviz
import itertools
import scipy
import pprint as pp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from spacy import displacy
from nltk.corpus import verbnet
from xml.etree import ElementTree

########## GLOBALS ###########

useConstituency = True

linearTextFlag = True

# T5TokenizerFast
#arg_constraints = {}
#validate_args=False

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# language
language = 'en'

# set up
nlp = spacy.load('en_core_web_md')
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

### semantic methods
#method = 'verbnet'
#method = 'wiktionary'

## wiktionary
#gzTool = "/usr/local/bin/gztool"
#wiktDataFile = os.path.expanduser("~/work/data/wiktionary/raw-wiktextract-data.json.gz")
#wiktIndexFile = os.path.expanduser("~/work/data/wiktionary/wiktextract_index.pkl")
##wiktFields = ['word', 'lang', 'pos', 'senses']

prepTags = ['ADV', 'ADVP', 'ADP', 'IN', 'PRT']

spaceOrPunct = ['SPACE', 'PUNCT']

importantPos = ['VERB', 'NOUN', 'ADJ', 'PRON']

notSpellings = ["n't", "not"]

########## FUNCTIONS ###########

def intListMax(a):
    if len(a) == 0:
        return 0
    else:
        return max(a)

def list2csv(lst, sep = ','):
    return sep.join(lst)

# remove item from lst [in-place]
def listRemove(lst, item):
    try:
        idx = lst.index(item)
        lst[:] = lst[:idx] + lst[idx+1:]
    except ValueError:
        pass
    return

# replace item with replacer in lst [in-place] (taken from some StackExch page)
def listReplace(lst, item, replacer):
    # fast and readable
    base=0
    for cnt in range(lst.count(item)):
        offset=lst.index(item, base)
        lst[offset]=replacer
        base=offset+1
    return

# replace j with i in graph dict [in-place] 
def replaceInGraph(gph, v, u):
    for i in gph:
        if v in gph[i][1]:
            listReplace(gph[i][1], v, u)
    return #gph

# return list of all nodes in adjacency lists of graph
def nodesInAdjLists(gph):
    nodes = set()
    for i in gph:
        for j in gph[i][1]:
            nodes.add(j)
    return nodes

# remove space/punctuation nodes [in-place]
def removeSpaceNodes(gph):
    delete = set()
    for i in gph:
        toki = gph[i][0][0]
        posi = gph[i][0][2]
        if posi in spaceOrPunct:
            delete.add(i)
            for j in gph:
                listRemove(gph[j][1], i)
    for d in delete:
        gph.pop(d, None)
    return #gph

# contract like nodes in graph [in-place]
def contractLikeNodes(gph):
    delete = set()
    for i in gph:
        toki = gph[i][0][0]
        lemi = gph[i][0][1]
        posi = gph[i][0][2]
        for j in gph:
            if i < j: # "like nodes" is a symmetric condition
                tokj = gph[j][0][0]
                lemi = gph[j][0][1]
                posj = gph[j][0][2]
                if lemi == lemj:                 # "like nodes" = same lemma
                    delete.add(j)                # mark node j for deletion
                    gph[i][1].extend(gph[j][1])  # add j's adjlist to i's
                    replaceInGraph(gph, j, i)    # replace j with i everywhere
    for d in delete:
        gph.pop(d, None)
    return #gph

# return the adjacency matrix of the graph
def adjacencyMatrix(gph):
    node2idx = dict() # node ID to matrix indices
    idx2node = dict() # inverse mapping
    nodes = list(gph.keys())
    n = len(nodes)
    A = np.zeros((n,n))
    i = 0
    for u in gph:
        node2idx[u] = i
        idx2node[i] = u
        j = 0
        for v in gph:
            if v in gph[u][1]: # orig trees were dir, now want undir, no "u<v"
                A[i,j] = 1.0
                A[j,i] = 1.0
            j += 1
        i += 1
    return A, node2idx, idx2node

# return nodes on path from i to j in predecessor matrix P
def path(P,i,j):
    s = j
    p = [s]
    while (s := P[i,s]) != i:
        p.append(s)
    p.append(i)
    p.reverse()
    return p

# project gph on subset of nodes, paths over del'd nodes become edges [in-place]
def projectOnNodes(gph, nodes):
    # find word and not-word nodes
    keep = set(nodes)
    delete = set(gph.keys()).difference(keep)
    # compute all shortest paths
    A, n2i, i2n = adjacencyMatrix(gph)
    D,P = scipy.sparse.csgraph.floyd_warshall(A, return_predecessors=True)
    # collapse paths over deleted nodes to edges on kept nodes
    for u in keep:
        for v in keep:
            if u<v:
                # find shortest path
                p = [gph[i2n[j]][0][0].text for j in path(P,n2i[u],n2i[v])]
                if len(p) > 2:
                    # edge not already there, add it
                    gph[u][1].append(v)
        gph[u][1][:] = list(set(gph[u][1]))
    # delete from adj lists
    for u in keep:
        # remove deleted nodes
        for d in delete:
            try:
                gph[u][1].remove(d)
            except ValueError:
                pass
    # delete dict keys
    for d in delete:
        gph.pop(d, None)
    return 

# project graph on word nodes, paths over del'd nodes become edges [in-place]
def projectOnWordNodes(gph):
    # find word nodes
    keep = set()
    for u in gph:
        toku = gph[u][0][0]
        if len(toku.text.split()) == 1:
            keep.add(u)
    projectOnNodes(gph, keep)
    return 

# project graph on word nodes with given Pos [in-place]
def projectOnPosNodes(gph, acceptPos=importantPos):
    keep = set()
    for u in gph:
        toku = gph[u][0][0]
        if len(toku.text.split()) == 1:
            if toku[0].pos_ in acceptPos:
                keep.add(u)
    projectOnNodes(gph, keep)
    return 
    

# aggregate edges, return aggregation cardinality as weight edge vector
def aggregateEdges(gph):
    weight = dict()
    for u in gph:
        adju = gph[u][1]
        for v in adju:
            uvw = adju.count(v)
            weight[(u,v)] = uvw
        gph[u][1][:] = list(set(adju))
    return weight

# write graphviz representation of graph to file
def graphViz(gph, outf, edgew=None, name="my graph", directed=False):
    g = None
    if directed:
        g = graphviz.Digraph(comment=name, filename=outf)
    else:
        g = graphviz.Graph(comment=name, filename=outf)
    for v in gph:
        nlab = gph[v][0][0].text.strip('\n')
        if len(nlab) > 0:
            g.node(str(v), nlab)
    for u in gph:
        for v in gph[u][1]:
            if edgew is not None and (u,v) in edgew:
                g.edge(str(u), str(v), label=str(edgew[(u,v)]))
            else:
                g.edge(str(u), str(v), label=gph[v][0][2]) # POS field
    g.view()
    return g

# return verbs in a sentence
def verbs(s):
    return [t.lemma_ for t in s if t.pos_ == 'VERB']

# query the wiktionary
def wiktQuery(query, wkindex):
    result = []
    if query in wkindex:
        lines = wkindex[query]
        #print("{}: query '{}' on line(s) {}".format(cmdname, query, lines))
        for l in lines:
            cmd = [gzTool, '-L', str(l), '-R', '1', wiktDataFile]
            out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.decode('utf-8')
            try:
                rec = json.loads(out)
                #rec = eval(out)
                #rec = ast.literal_eval(out)
                #rec = ast.parse(out, mode='eval')
            except:
                print(out)
                print("wiktQuery:error: the above content could not be parsed")
                quit()
            result.append(rec)
    return result

# recursive exploration of json tree
# from https://stackoverflow.com/questions/21028979/how-to-recursively-find-specific-key-in-nested-json
def jsonFind(json_input, lookup_key):
    if isinstance(json_input, dict):
        for k, v in json_input.items():
            if k == lookup_key:
                yield (k,v)
            else:
                yield from jsonFind(v, lookup_key)
    elif isinstance(json_input, list):
        for item in json_input:
            yield from jsonFind(item, lookup_key)
    return

# return phrasal verbs
def phrasalVerbs(parseNode, ret=None):
    # at this node
    if ret is None:
        ret = []
    labels = parseNode._.labels
    for l in labels:
        # check if current node is verb phrase and subtree has preposition
        if l == 'VP' and any(containsTag(parseNode, prep) for prep in prepTags):
            theVerb = findFirstByTag(parseNode, 'VERB')
            if theVerb is not None:
                for tok in findPrepositions(parseNode):
                    ret.append((theVerb, tok))
    # recursive part
    for n in parseNode._.children:
        phrasalVerbs(n, ret)
    return ret

# true if node is a leaf node in a constituency tree
def isLeaf(parseNode):
    if len(list(parseNode._.children)) == 0:
        return True
    return False

# returns first token with given tag in node (as a sentence)
def findFirstByTag(parseNode, tag):
    for token in parseNode:
        if token.pos_ == tag or token.tag_ == tag:
            return token
    return None

# returns first token with given tag in node (as a sentence)
def findPrepositions(parseNode):
    return [token for token in parseNode if token.pos_ in prepTags or token.tag_ in prepTags]

# returns true if tag appears in some label in constituency subtree
def containsTag(parseNode, tag):
    # at this node
    lab = parseNode._.labels
    if len(lab) == 0:
        # only at leaf nodes: check if tag === token POS tag
        if tag == parseNode[0].pos_:
            return True
    # recursive part
    for n in parseNode._.children:
        labels = n._.labels
        for l in labels:
            # check if tag is one of the constituency labels
            if tag == l:
                return True
    return False

# clean up the sentence text from newlines, which impact graphviz's comment
def cleanSentence(s):
    return s.replace('\n', ' ')

# explore constituency Tree from benepar in spacy
def constituencyPrint(node, level):
    tab = '  '
    # at this node
    lab = node._.labels
    if len(lab) == 0:
        print(tab*level + node[0].pos_ + '/' + node[0].tag_, ':', node.text)
    else:
        print(tab*level, end='')
        for l in lab:
            print(l, end=' ')
        print(':', node.text)
    # recursive part
    for n in node._.children:
        if n[0].pos_ not in spaceOrPunct:
            constituencyPrint(n, level+1)
    return


# explore dependency tree
def dependencyPrint(node, level):
    tab = '  '
    # at this node
    print(tab*level + node.pos_ + '/' + node.tag_, ':', node.text)
    # recursive part
    for n in node.children:
        if n.pos_ not in spaceOrPunct:
            dependencyPrint(n, level+1)
    return

# constituency tree graph encoded in dict
def constituencyGraph(node, gph):
    ## recursive part
    subnodes = []
    for n in node._.children:
        if n[0].pos_ not in spaceOrPunct:
            subnodes.append(constituencyGraph(n, gph))
    ## at this node
    lab = node._.labels
    if len(lab) == 0:
        nodelabel = [node, node[0].lemma_, node[0].pos_, node[0].tag_]
    else:
        nodelabel = [node, node.lemma_]
        for l in lab:
            nodelabel.append(l)
    id = len(gph.keys()) + 1
    gph[id] = (nodelabel, subnodes)
    return id

# dependency tree graph encoded in dict
def dependencyGraph(node, gph):
    ## recursive part
    subnodes = []
    for n in node.children:
        if n.pos_ not in spaceOrPunct:
            subnodes.append(dependencyGraph(n, gph))
    ## at this node
    # find new node ID
    id = intListMax(gph.keys()) + 1
    nodelabel = [node, node.lemma_, node.pos_, node.tag_]
    gph[id] = (nodelabel, subnodes)
    return id

# compute syntax trees (as NX graphs) for sentences in a document
#   separate tree graphs have nodes with disjoint IDs (startID is different)
def syntaxTrees(doc, conFlag):
    T = dict()
    startID = 1
    for i,sent in enumerate(doc.sents):
        # get tree as dict graph Gd
        Gd = dict()                     
        if conFlag:
            constituencyGraph(sent, Gd)
        else:
            dependencyGraph(sent.root, Gd)
        # encode in NX graph
        T[i] = nx.DiGraph()
        T[i].add_nodes_from(Gd.keys())  # NX nodes
        labels = {v:Gd[v][0][0] for v in Gd} # labels
        lem = {v:Gd[v][0][1] for v in Gd}    # lemmata
        pos = {v:Gd[v][0][2] for v in Gd}    # POS
        tag = {}                             # TAG (not always there)
        for v in Gd:
            try:
                tag[v] = Gd[v][0][3]
            except:
                tag[v] = ""
        nx.set_node_attributes(T[i], labels, "labels")
        nx.set_node_attributes(T[i], lem, "lem")
        nx.set_node_attributes(T[i], pos, "pos")
        nx.set_node_attributes(T[i], tag, "tag")
        for v in T[i].nodes:  # NX edges
            T[i].add_edges_from(list((v,u) for u in Gd[v][1]), weight=1.0)
        startID += T[i].number_of_nodes()  # update startID
    return T

# draw a syntax tree; block=False for many windows (last win must block though)
def drawTree(Ti, block=True, labelName="labels"):
    plt.figure(1)
    x = graphviz_layout(Ti, prog='dot')
    #x = nx.rescale_layout_dict(x, scale=3)
    labels = nx.get_node_attributes(Ti, labelName)
    nx.draw(Ti, x, labels=labels, with_labels=True, node_color="none", bbox=dict(facecolor="skyblue", edgecolor='black', boxstyle='round,pad=0.1'))
    plt.show(block=block)

# draw a graph-of-words; block=False for multiple windows (last one must block)
def drawGraph(G, block=True, labelName="labels"):
    plt.figure(2)
    x = graphviz_layout(G, prog='circo')
    #x = nx.rescale_layout_dict(x, scale=3)
    labels = nx.get_node_attributes(G, labelName)
    nx.draw(G, x, labels=labels, with_labels=True, node_color="none", bbox=dict(facecolor="skyblue", edgecolor='black', boxstyle='round,pad=0.1'))
    w = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, x, edge_labels=w)
    plt.show(block=block)

# shortest path on a tree givent the lowest common ancestor (LCA)
def shortestPathOnTree(Ti, u, v, lca):
    next = u
    p = [next]
    while next != lca:
        next = list(Ti.predecessors(next))[0]
        p.append(next)        
    next = v
    while next != lca:
        p.append(next)        
        next = list(Ti.predecessors(next))[0]
    return len(p)-1, p

# token graph from syntax tree:
# contract all non-leaf nodes of a single sintax tree to a weighted graph:
# - compute all shortest-path distances (based on LCA) on leaves
# - create new edges on leaves for shortpaths on trees (pathlen = edge weight)
# NOTE: edge weight is a syntactical distance
def tokenGraph(Ti):
    lab = nx.get_node_attributes(Ti, "labels")
    lem = nx.get_node_attributes(Ti, "lem")
    pos = nx.get_node_attributes(Ti, "pos")
    tag = nx.get_node_attributes(Ti, "tag")
    # compute leaf nodes
    leaves = [v for v in Ti.nodes if Ti.out_degree(v) == 0]
    # all pairs of leaf nodes
    pairs = list(itertools.combinations(leaves, 2))
    # all LCAs on leaf pairs
    r = max(Ti.nodes)
    lca = dict(nx.tree_all_pairs_lowest_common_ancestor(Ti,root=r,pairs=pairs))
    # new weighted graph
    wG = nx.Graph()
    wG.add_nodes_from(leaves)
    nx.set_node_attributes(wG, {k:lab[k] for k in leaves}, "labels")
    nx.set_node_attributes(wG, {k:lem[k] for k in leaves}, "lem")
    nx.set_node_attributes(wG, {k:pos[k] for k in leaves}, "pos")
    nx.set_node_attributes(wG, {k:tag[k] for k in leaves}, "tag")    
    for (u,v),l in lca.items():
        c,p = shortestPathOnTree(Ti, u,v,l)
        wG.add_edge(u,v, weight=c)
    return wG

# create a graph of words (conFlag=True constituency else dependency):
# 1. a path from first to last token of doc (edge weight=1)
# 2. weighted edges from all of the syntax trees projected on leaves
# 3. contract nodes with same token text
# If linearTextFlag=True add a Hamiltonian path in the linear text order
def graphOfWords(doc, conFlag, lintext=linearTextFlag):
    # compute all syntax trees
    T = syntaxTrees(doc, conFlag)
    # construct dictionary of token graphs for each sentence
    Gs = dict()
    startID = 0
    for i in T:
        Gs[i] = tokenGraph(T[i])
        # relabel the vertices so that the union is vertex disjoint
        rn = {v:v+startID for v in Gs[i].nodes}
        nx.relabel_nodes(Gs[i], rn, copy=False)
        startID = max(Gs[i].nodes)
        #drawTree(T[i], block=False)
        #drawGraph(Gs[i])
    # collect all node properties from single graphs into a union
    allnodes = []
    labels = dict()
    lem = dict()
    pos = dict()
    tag = dict() 
    for i in Gs:
        allnodes.extend(list(Gs[i].nodes))
        labels = labels | nx.get_node_attributes(Gs[i], "labels")
        lem = lem | nx.get_node_attributes(Gs[i], "lem")
        pos = pos | nx.get_node_attributes(Gs[i], "pos")
        tag = tag | nx.get_node_attributes(Gs[i], "tag")
    # create the disjoint union of all of the sentence graphs
    G = nx.Graph()
    G.add_nodes_from(allnodes)
    nx.set_node_attributes(G, labels, "labels")
    nx.set_node_attributes(G, lem, "lem")
    nx.set_node_attributes(G, pos, "pos")
    nx.set_node_attributes(G, tag, "tag")
    for i in Gs:
        G.add_weighted_edges_from(Gs[i].edges.data('weight'))
    # optional linear text order
    if lintext:
        # add a path encoding the linear order of the text
        # NOTE: the recursive indexing in constituency/dependencyTree functions
        #       ensure that node IDs are order-isomorphic with linear text order
        u = allnodes[0]
        for v in allnodes[1:]:
            if v in G.neighbors(u):
                # change weight to 1.0 (closeness due to linear text order)
                G[u][v]['weight'] = 1.0
            else:
                G.add_edge(u,v,weight=1.0)
            u = v
    # contract like nodes using equivalence classes
    def likeNodes(u,v):
        eqrel = (lem[u] ==lem[v]) or (labels[u].text in notSpellings and labels[v].text in notSpellings)
        #eqrel = (labels[u].text==labels[v].text and pos[u]==pos[v]) or (labels[u].text in notSpellings and labels[v].text in notSpellings)
        return eqrel
    E = nx.equivalence_classes(allnodes, likeNodes)
    for ec in E:
        C = sorted(list(ec))
        u = C[0]
        for v in C[1:]:
            nx.contracted_nodes(G, u,v, self_loops=False, copy=False)
    return G
    
########### MAIN ############

if len(sys.argv) < 2:
    print("syntax is: {} textfile [constituency|dependency]".format(sys.argv[0]))
    print("  default is constituency")
    exit(1)

# # read index
# with open(wiktIndexFile, 'rb') as wf:
#     wiktIdx = pickle.load(wf)

# read text
with open(sys.argv[1], 'r') as tf:
    txt = tf.read()

if len(sys.argv) >= 3:
    if sys.argv[2].startswith('dep'):
        useConstituency = False

# parse the text
doc = nlp(txt)
G = graphOfWords(doc,useConstituency)
drawGraph(G, block=True, labelName="lem")


