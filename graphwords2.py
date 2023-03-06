#!/usr/bin/env python3.10

## graph of words where the sliding window on the linear word order
##   is replaced by a syntax tree (either constituency or dependency)
## Leo Liberti
## work started 230301

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


## method for creating Graph of Words
gowMethod = "constituency"
gowClassicRadius = 4

## draw the graph?
drawGraphFlag = False
#drawGraphFlag = True

## whether to add a linear text path to tree-based methods
linearTextFlag = True
linearTextFlag = False

## fewer warnings at launch (from T5TokenizerFast?)
arg_constraints = {}
validate_args=False
os.environ["TOKENIZERS_PARALLELISM"] = "false"

## set spacy up
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

## various useful categories of word tags
spaceOrPunct = ['SPACE', 'PUNCT']
removeTags = ['HYPH']
importantPos = ['VERB', 'NOUN', 'ADJ', 'VP']
#importantPos = ['VERB', 'NOUN', 'ADJ', 'PRON']
notSpellings = ["n't", "not"]
prepTags = ['ADV', 'ADVP', 'ADP', 'IN', 'PRT']
genericLemmata = ["do", "be"]

########## FUNCTIONS ###########

def intListMax(a):
    if len(a) == 0:
        return 0
    else:
        return max(a)

# return nodes on path from i to j in predecessor matrix P
def path(P,i,j):
    s = j
    p = [s]
    while (s := P[i,s]) != i:
        p.append(s)
    p.append(i)
    p.reverse()
    return p


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
        if n[0].pos_ not in spaceOrPunct and n[0].tag_ not in removeTags:
            constituencyPrint(n, level+1)
    return


# explore dependency tree
def dependencyPrint(node, level):
    tab = '  '
    # at this node
    print(tab*level + node.pos_ + '/' + node.tag_, ':', node.text)
    # recursive part
    for n in node.children:
        if n.pos_ not in spaceOrPunct and n.tag_ not in removeTags:
            dependencyPrint(n, level+1)
    return

# draw a syntax tree; block=False for many windows (last win must block though)
def drawTree(Ti, block=True, labelName="labels", engine='dot', pltId=1):
    plt.figure(pltId)
    x = graphviz_layout(Ti, prog=engine)
    #x = nx.rescale_layout_dict(x, scale=3)
    labels = nx.get_node_attributes(Ti, labelName)
    nx.draw(Ti, x, labels=labels, with_labels=True, node_color="none", bbox=dict(facecolor="skyblue", edgecolor='black', boxstyle='round,pad=0.1'))
    plt.show(block=block)

# draw a graph-of-words; block=False for multiple windows (last one must block)
def drawGraph(G, block=True, labelName="labels", engine='dot', pltId=2):
    plt.figure(pltId)
    x = graphviz_layout(G, prog=engine)
    #x = nx.rescale_layout_dict(x, scale=3)
    labels = nx.get_node_attributes(G, labelName)
    nx.draw(G, x, labels=labels, with_labels=True, node_color="none", bbox=dict(facecolor="skyblue", edgecolor='black', boxstyle='round,pad=0.1'))
    w = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, x, edge_labels=w)
    plt.show(block=block)

# constituency tree graph encoded in dict
def constituencyGraph(node, gph):
    ## recursive part
    subnodes = []
    for n in node._.children:
        if n[0].pos_ not in spaceOrPunct and n[0].tag_ not in removeTags:
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
        if n.pos_ not in spaceOrPunct and n.tag_ not in removeTags:
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
        if conFlag:
            # constituency trees have non-leaf nodes labelled by POS
            for v in Gd:
                if len(Gd[v][1]) > 0:
                    labels[v] = pos[v]
        nx.set_node_attributes(T[i], labels, "labels")
        nx.set_node_attributes(T[i], lem, "lem")
        nx.set_node_attributes(T[i], pos, "pos")
        nx.set_node_attributes(T[i], tag, "tag")
        for v in T[i].nodes:  # NX edges
            T[i].add_edges_from(list((v,u) for u in Gd[v][1]), weight=1.0)
        startID += T[i].number_of_nodes()  # update startID
    return T

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

# leaf node graph from constituency tree:
# contract all non-leaf nodes of a single sintax tree to a weighted graph:
# - compute all shortest-path distances (based on LCA) on leaves
# - create new edges on leaves for shortpaths on trees (pathlen = edge weight)
# NOTE: edge weight is a syntactical distance
def constituencyLeafGraph(Ti):
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

# remove nodes with given prop, turn 2-paths through removed nodes with edges
# from <https://stackoverflow.com/questions/53353335/networkx-remove-node-and-reconnect-edges>
# NOTE: this is not in-place
def removeAndReconnect(G:nx.Graph, removal_pred:callable):
    g = G.copy()
    while any(removal_pred(v) for v in g.nodes):
        g0 = g.copy()
        for u in g.nodes:
            if removal_pred(u):
                edges_containing_node = g.edges(u)
                dst_to_link = [e[1] for e in edges_containing_node]
                dst_p_to_link = list(itertools.combinations(dst_to_link,r=2))
                for p in dst_p_to_link:
                    d = nx.shortest_path_length(g0,p[0],p[1],weight='weight')
                    g0.add_edge(p[0], p[1], weight=d)
                g0.remove_node(u)
                break
        g = g0
    return g

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
        if conFlag:
            Gs[i] = constituencyLeafGraph(T[i])
        else:
            Gs[i] = T[i]
        #drawTree(T[i])
        # relabel the vertices so that the union is vertex disjoint
        rn = {v:v+startID for v in Gs[i].nodes}
        nx.relabel_nodes(Gs[i], rn, copy=False)
        startID = max(Gs[i].nodes)
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
    def likeNodes(u,v): return lem[u] == lem[v]
    E = nx.equivalence_classes(allnodes, likeNodes)
    for ec in E:
        C = sorted(list(ec))
        u = C[0]
        for v in C[1:]:
            nx.contracted_nodes(G, u,v, self_loops=False, copy=False)
    # only keep nodes with important POS, reconnect around removed nodes
    G = removeAndReconnect(G, lambda v : pos[v] not in importantPos)
    return G

# traditional graph-of-words with sliding window of given radius
def classicGraphOfWords(doc, radius=2):
    # only keep interesting tokens
    toks = [t for t in doc if t.pos_ in importantPos and t.lemma_ not in genericLemmata]
    labels = {i+1:t.text for i,t in enumerate(toks)}
    lem = {i+1:t.lemma_ for i,t in enumerate(toks)} 
    # make the old-style graph-of-words
    G = nx.Graph()
    nx.add_path(G, list(labels.keys()), weight=1.0)
    nx.set_node_attributes(G, labels, "labels") 
    n = G.number_of_nodes()
    # add edges from sliding text window
    inner_nodes = [v for v in G.nodes if v-radius >= 1 and v+radius <= n]
    for v in inner_nodes:
        for i in range(v-radius,v+radius+1):
            for j in range(v-radius,v+radius+1):
                if j-i > 1:
                    # out of initial path
                    G.add_edge(i,j, weight=j-i)
                    #print("adding ({},{},w={})".format(i,j,j-i))
    # contract like nodes using equivalence classes
    def likeNodes(u,v): return lem[u] == lem[v]
    E = nx.equivalence_classes(list(lem.keys()), likeNodes)
    for ec in E:
        C = sorted(list(ec))
        u = C[0]
        for v in C[1:]:
            #print("contract {} and {}".format(u,v))
            nx.contracted_nodes(G, u,v, self_loops=False, copy=False)
    return G
    

########### MAIN ############

if len(sys.argv) < 2:
    print("syntax is: {} textfile [constituency|dependency|classic=radius]".format(sys.argv[0]))
    print("  creates a graph-of-words from text in a textfile with various method")
    print("    graph-of-words vertices numbered from 1")
    print("  methods:")
    print("    constituency: word relations from constituency tree paths")
    print("    dependency:   word relations from dependency tree")
    print("    classic:      word relations from sliding window of given radius")
    print("    default is constituency")
    exit(1)

# # read index
# with open(wiktIndexFile, 'rb') as wf:
#     wiktIdx = pickle.load(wf)

# read text
with open(sys.argv[1], 'r') as tf:
    txt = tf.read()
txt.replace('\n',' ').replace('\r',' ')
    
# parse command line options
conFlag = True
if len(sys.argv) >= 3:
    if sys.argv[2].startswith('dep'):
        gowMethod = "dependency"
        conFlag = False
        print("{}: graph of words on dependency tree".format(sys.argv[0]))
    elif sys.argv[2].startswith('classic='):
        gowMethod = "classic"
        rad = int(sys.argv[2].split('=')[1])
        if rad < 1:
            print("{}: classic method chosen with invalid radius={}<1, abort".format(sys.argv[0], rad))
            exit(2)
        gowClassicRadius = rad
        print("{}: graph of words on sliding text window of radius={}".format(sys.argv[0], gowClassicRadius))
    else:
        gowMethod = "constituency"
        print("{}: graph of words on constituency tree".format(sys.argv[0]))
        
# create the graph of words
doc = nlp(txt)
G = None
if gowMethod == "classic":
    G = classicGraphOfWords(doc, gowClassicRadius)
    lname = "labels"
    outName = '.'.join(os.path.basename(sys.argv[1]).split('.')[:-1]) + "-classic_r" + str(gowClassicRadius) + '.dot'
else:
    G = graphOfWords(doc, conFlag)
    lname = "labels" # or "lem" for lemmata
    outName = '.'.join(os.path.basename(sys.argv[1]).split('.')[:-1])
    if conFlag:
        outName += "-constituency.dot"
    else:
        outName += "-dependency.dot"
if drawGraphFlag:
    drawGraph(G, labelName=lname)
# save it to a graphViz .dot file
nx.drawing.nx_agraph.write_dot(G, outName)
