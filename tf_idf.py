#!/usr/bin/env python3.10

## graph-of-words based TF-IDF
##   (comparison with Rousseau & Vazirgiannis)
## Leo Liberti
## work started 230302

########### IMPORTS ###########
import sys
import os
import math
import json
import gzip
import pickle
import pprint
import ast
import graphviz
import itertools
import scipy
import glob
import re
import pprint as pp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

########## GLOBALS ###########

## show partial results for each measure?
showPartialResults = False
showPartialResults = True

# show up to max ranking
maxRankShow = None # no maximum
maxRankShow = 6

# show rank comparison
showRankComparison = False

# show BM25?
showBM25 = False

# show TFIDF?
showTFIDF = False

## draw the graph?
drawGraphFlag = False
#drawGraphFlag = True

## the magic k1 constant (Rousseau & Vazirgiannis, after Eq (4))
magic_k1 = 1.2

## how does one choose b in TFp? must be in [0,1]...
magic_b = 0.5

## again a magic value
magic_delta = 1.0

## average document length in corpus (set manually)
avgDocLenInCorpus = 4

## standard string replacements
stringReplacements = {"’":"'", "…":"...", '“':'"', '”':'"'}

########## FUNCTIONS ###########

# raw term frequency: term, documents are strings
def tf(term, document):
    occurrences = len(re.findall(r"\b" + re.escape(term) + r"\b", document))
    return occurrences

# relative term frequency: t is the term, d is the document, both are strings
def rel_tf(t,d):
    numer = tf(t, d)
    denom = 0
    terms = d.split()
    for tm in terms:
        term = re.sub('[\W_]+', '', tm)
        denom += tf(term, d)
    return numer / denom

# concave term frequency: t is the term, d is the document, both are strings
def TFl(t,d):
    tftd = tf(t,d)
    if tftd == 0:
        return 1
    else:
        return 1 + math.log(1 + math.log(tf(t,d)))

# pivoted normaliz weight: t is the term, d is the document, both are strings
def pivotedNormalizationWeighting(t,d, b=magic_b, avdl=avgDocLenInCorpus):
    return TFl(t,d) / (1 - b + b*(len(d.split()) / avdl))

# IDF: t=term, C=corpus (a dict filename:string)
def IDF(t, C):
    dft = sum(int(tf(t, C[d]) > 0) for d in C) # here d is an index to C
    if dft > 0:
        return math.log((len(C)+1) / dft)
    else:
        return math.log(len(C)+1)

# TFIDF: t=term (string), C=corpus, d=document (key to dict C)
def TFIDF(t,d,C):
    return pivotedNormalizationWeighting(t,C[d]) * IDF(t,C)

# BM25: t=term (string), C=corpus, d=document (key to dict C)
def BM25(t,d,C, k1=magic_k1, b=0.75, avdl=avgDocLenInCorpus):
    K = k1*(1-b+b*(len(C[d].split()) / avdl))
    bm25 = (((k1+1)*tf(t,C[d])) / (K+tf(t,C[d])))*IDF(t,C)
    return bm25

# TW-IDF (graph-of-words):
#   t=int, term=dict(int:str), C=dict, d=int(C key), G=nx.Graph
def TWIDF(t,term,d,C,G, b=0.75, avdl=avgDocLenInCorpus):
    tm = term[t]
    tw = sum(float(G[t][v]['weight']) for v in G[t])
    twidf = tw*IDF(tm,C) / (1-b+b*len(C[d].split())/avdl)
    return twidf

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

########### MAIN ############

cmd = sys.argv[0]

if len(sys.argv) < 2:
    print("syntax is: {} doc [gow_spec]".format(cmd))
    print("  doc is the basename of a filename in a corpus directory")
    print("  there must exist two files, doc.txt and doc.dot, the corpus dir:")
    print("    doc.txt is the document")
    print("    doc.dot is the corresponding graph-of-words made by graphwords2.py")
    print("  if gow_spec is given, the graph-of-words file name is doc-gow_spec.dot")
    exit(1)

## parse command line

# doc name without extension (as required on cmd line)
doc = os.path.basename(sys.argv[1])
if doc[-1] == '.': # typical errors with filename expansions
    doc = doc[:-1]
# doc filename without dir prefix
docfn = doc + ".txt"
# corpus dir name
dir = os.path.dirname(sys.argv[1])
# all file names in dir
textFilesInCorpus = glob.glob(dir+"/*.txt")
# dict containing corpus
C = dict()
# fill corpus dict
for txt in textFilesInCorpus:
    with open(txt, "r") as f:
        # keys are filenames (with extension) without dir
        fn = os.path.basename(txt)
        # translate weird characters to ASCII-128
        C[fn] = f.read().translate(str.maketrans(stringReplacements))
if docfn not in C:
    print("{}: error: doc file {} not found in corpus {}".format(cmd,docfn, dir))
    exit(2)

# read graph-of-words
dotfn = doc + ".dot"
gowSpec = None
if len(sys.argv) >= 3:
    gowSpec = sys.argv[2]
    if gowSpec.startswith("con"):
        gowSpec = "constituency"
    elif gowSpec.startswith("dep"):
        gowSpec = "dependency"
    elif gowSpec.startswith("cl"):
        gowSpec = "classic_r"
        templ = dir + "/" + doc + "-" + gowSpec + "*.dot"
        compl = glob.glob(templ)
        if len(compl) == 0:
            print("{}: error: can't find any file {}".format(cmd,templ))
            exit(4)
        gowSpec += compl[0].split('_r')[1].split('.')[0]
    dotfn = doc + "-" + gowSpec + ".dot"
dotpath = dir + "/" + dotfn
try:
    G = nx.Graph(nx.nx_pydot.read_dot(dotpath))
except FileNotFoundError:
    # read first eligible file
    eligibles = glob.glob(dir + "/" + doc + "-*.dot")
    if len(eligibles) > 0:
        dotpath = eligibles[0]
        print("{}: warning: reading graph from {}".format(cmd, dotpath))
        G = nx.Graph(nx.nx_pydot.read_dot(dotpath))
    else:
        print("{}: error: .dot file not found for doc {} in dir {}".format(cmd, doc, dir))
        exit(3)
if drawGraphFlag:
    drawGraph(G)
print(C[docfn])

# read terms
term = nx.get_node_attributes(G, "labels")
pp.pprint(term)

# average document length in corpus
avgDocLenInCorpus = np.mean(np.array([len(C[fn].split()) for fn in C]))

# compute TF-IDF
if showPartialResults and showTFIDF:
    print("{}: TF-IDF ===================".format(cmd))
tfidf = dict()
for t in term:
    tm = term[t]
    tfidf[tm] = TFIDF(tm,docfn,C)
tfidf_rank = sorted([(v,k) for k,v in tfidf.items()], reverse=True)
if showPartialResults and showTFIDF:
    print("rank,term,tfidf")
for i,(v,k) in enumerate(tfidf_rank):
    tfidf[k] = (tfidf[k],i+1)
    if showPartialResults and showTFIDF:
        if maxRankShow is None or i < maxRankShow:
            print("{:d},{:s},{:g}".format(i+1,k,v))
    
# compute BM25
if showPartialResults and showBM25:
    print("{}: BM25 =====================".format(cmd))
bm25 = dict()
for t in term:
    tm = term[t]
    bm25[tm] = BM25(tm,docfn,C)
bm25_rank = sorted([(v,k) for k,v in bm25.items()], reverse=True)
if showPartialResults and showBM25:
    print("rank,term,bm25")
for i,(v,k) in enumerate(bm25_rank):
    bm25[k] = (bm25[k],i+1)
    if showPartialResults and showBM25:
        if maxRankShow is None or i < maxRankShow:
            print("{:d},{:s},{:g}".format(i+1,k,v))
    
# compute (graph-of-words based) TW-IDF
if showPartialResults:
    print("{}: TW-IDF ({}) ======".format(cmd, gowSpec))
twidf = dict()
for t in term:
    tm = term[t]
    twidf[tm] = TWIDF(t,term,docfn,C,G)
twidf_rank = sorted([(v,k) for k,v in twidf.items()], reverse=True)
if showPartialResults:
    print("rank,term,twidf")
for i,(v,k) in enumerate(twidf_rank):
    twidf[k] = (twidf[k],i+1)
    if showPartialResults:
        if maxRankShow is None or i < maxRankShow:
            print("{:d},{:s},{:g}".format(i+1,k,v))

# put together all ranks for comparison
if showRankComparison:
    print("{}: rank comparison ==========".format(cmd))
compare = dict()
for t in term:
    tm = term[t]
    compare[t] = (tm,tfidf[tm][1],bm25[tm][1],twidf[tm][1])
if showRankComparison:
    print("term,tfidfrk,bm25rk,twidfrk")
    for v in compare.values():
        print("{},{},{},{}".format(v[0],v[1],v[2],v[3]))
