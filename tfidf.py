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

# show up to max ranking
maxRankShow = None # no maximum
maxRankShow = 5

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
def TFIDF(t,d,C,G=None):
    return pivotedNormalizationWeighting(t,C[d]) * IDF(t,C)

# BM25: t=term (string), C=corpus, d=document (key to dict C)
def BM25(t,d,C, G=None, k1=magic_k1, b=0.75, avdl=avgDocLenInCorpus):
    K = k1*(1-b+b*(len(C[d].split()) / avdl))
    bm25 = (((k1+1)*tf(t,C[d])) / (K+tf(t,C[d])))*IDF(t,C)
    return bm25

# TW-IDF (graph-of-words):
#   t=term (string), term=dict(int:str), C=dict, d=int(C key), G=nx.Graph
def TWIDF(t,d,C,G, b=0.75, avdl=avgDocLenInCorpus):
    terms = nx.get_node_attributes(G, "labels")
    invterms = {tm:i for i,tm in terms.items()}
    if t in invterms:
        u = invterms[t]
        tw = sum(float(G[u][v]['weight']) for v in G[u])
        twidf = tw*IDF(t,C) / (1-b+b*len(C[d].split())/avdl)
    else:
        twidf = 0.0
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
    return

# compute ranking from rank function
def computeRanking(typeName, docName, terms, corpus, graph):
    rkfun = dict()
    rank = dict()
    for t in terms:
        tm = terms[t]
        if typeName == "TFIDF":
            rkfun[tm] = TFIDF(tm, docName, corpus)
        elif typeName == "BM25":
            rkfun[tm] = BM25(tm, docName, corpus)
        elif typeName == "TWIDF":
            rkfun[tm] = TWIDF(tm, docName, corpus, graph)
        else:
            print("computeRanking: typeName={} not in [TFIDF,BM25,TWIDF]".format(typeName))
            exit(5)
        makerank = sorted([(v,k) for k,v in rkfun.items()], reverse=True)
        for i,(v,k) in enumerate(makerank):
            rank[i+1] = (k,v)
    return rank

# display ranking
def displayRanking(rank, methodName, maxRankShow):
    print("{}:".format(methodName))
    #print("rank,term,rkfunval")
    for k in rank:
        (t,v) = rank[k]
        if maxRankShow is None or k <= maxRankShow:
            print("{},{},{}".format(k,t,v))
    return

    
########### MAIN ############

cmd = sys.argv[0]

if len(sys.argv) < 2:
    print("syntax is: {} <dir|doc>".format(cmd))
    print("  dir is a corpus directory (batch mode)")
    print("  doc is the basename of a filename in dir (single document mode)")
    print("  there must exist files doc.txt and doc-*.dot in the corpus dir:")
    print("    doc.txt is the document")
    print("    doc-*.dot are the corresponding graph-of-words made by graphwords2.py")
    exit(1)

## parse command line
arg = sys.argv[1]
docMode = True
showCompOnly = False
if os.path.isdir(arg):
    docMode = False
    showCompOnly = True
dir = arg   # batch mode
if docMode: # doc mode
    # single document mode:
    # doc name without extension (as required on cmd line)
    doc = os.path.basename(arg)
    doc = os.path.splitext(doc)[0]
    # doc filename without dir prefix
    docfn = doc + ".txt"
    # corpus dir name
    dir = os.path.dirname(sys.argv[1])

## all file names in dir
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
if docMode and docfn not in C:
    print("{}: doc file {} not found in corpus {}".format(cmd,docfn, dir))
    exit(2)

## average document length in corpus
avgDocLenInCorpus = np.mean(np.array([len(C[fn].split()) for fn in C]))
    
## list of files to go through
docList = None
if docMode:
    # doc mode, just one file
    docList = [docfn] 
else:
    # or batch mode, all files in dir
    docList = list(C.keys())

## read ground truth if present
gtlist = glob.glob(dir + '/GROUNDTRUTH*.lst')
groundTruthFn = None
if len(gtlist) == 0:
    print("{}: no GROUNDTRUTH*.lst file found in {}".format(cmd, dir))
elif len(gtlist) > 1:
    groundTruthFn = gtlist[0]
    print("{}: many ground truth files found in {}, using {}".format(cmd, dir, os.path.splitext(groundTruthFn)[0]))
else:
    groundTruthFn = gtlist[0]
    print("{}: using ground truth file {}".format(cmd, dir, os.path.splitext(groundTruthFn)[0]))
GT = dict()
with open(groundTruthFn, 'r') as gt:
    for line in gt:
        l = line.split()
        l0 = os.path.splitext(os.path.basename(l[0]))[0]
        GT[l0] = {i:l[i] for i in range(1,len(l))}

## main loop
# ranking functions by document and method
rank = dict()
# keywords in GT by document and method
foundKw = dict()
# list of filenames (with extensions)
docList = sorted(docList)
# technicality to sort method names in dict
sortkeylen = 2
# start loop
testLimit = 500
for fileCounter,docfn in enumerate(docList):
    # docfn is a filename without prefix dirname
    # doc is docfn without the extension
    doc = os.path.splitext(docfn)[0]

    if fileCounter >= testLimit:
        break
    
    # read graphs-of-words
    templ = dir + "/" + doc + "-*.dot"
    compl = glob.glob(templ) # find possible .dot completions
    if len(compl) == 0:
        # no .dot files found for 
        print("{}: can't find any .dot file {}".format(cmd,templ))
        exit(5)
    gowSpec = [c.split(doc+'-')[1].split('.')[0] for c in compl]
    G = dict()
    for spec in gowSpec:
        dotfn = doc + "-" + spec + ".dot"
        dotwithpath = dir + "/" + dotfn
        G[spec] = nx.Graph(nx.nx_pydot.read_dot(dotwithpath))
        if drawGraphFlag:
            drawGraph(G[spec])

    # read terms for freq-based rank functions: union of terms in all GOWs
    termSet = set()
    for spec in gowSpec:
        tspec = nx.get_node_attributes(G[spec], "labels")
        termSet.update(tspec.values())
    term = {i+1:t for i,t in enumerate(termSet)}

    # compute term frequency based ranking functions
    tfidf = computeRanking("TFIDF", docfn, term, C, None)
    bm25 = computeRanking("BM25", docfn, term, C, None)

    # compute graph-of-words based ranking functions
    gow = dict()
    for spec in gowSpec:
        gow[spec] = computeRanking("TWIDF", docfn, term, C, G[spec])
        
    # collect rankings
    rank[doc] = {"00tfidf":tfidf, "01bm25":bm25}
    for c,spec in enumerate(gowSpec):
        keyspec = str(sortkeylen).zfill(2) + spec
        rank[doc][keyspec] = gow[spec]

    # print out results
    if maxRankShow is None:
        nshown = "all"
    else:
        nshown = min(maxRankShow, len(GT[doc].keys()))
    if nshown is None:
        nshown = "all"
    if showCompOnly and fileCounter == 0:
        print("OUTLABELS:file,nshown", end='')
        for method in sorted(rank[doc]):
            md = method[sortkeylen:]
            print(",{}".format(md), end='')
        print()
    if not showCompOnly:
        # print text
        print(doc, "======================")
        print(C[docfn])
    
        # print rankings
        print(doc, "rankings ({} terms shown):".format(nshown))    
        for method in sorted(rank[doc]):
            md = method[sortkeylen:]
            displayRanking(rank[doc][method], md, maxRankShow)
        #pp.pprint(rank[doc])
        if doc in GT:
            print("ground truth:")
            for k in GT[doc]:
                if maxRankShow is None or k <= maxRankShow:
                    print("{},{},{}".format(k, GT[doc][k], 1.0))
            #pp.pprint(GT[doc])
        else:
            print("warning: no ground truth found for {}".format(doc))

    # print comparison with ground truth (number of correct keywords)
    if not showCompOnly:
        print("OUTLABELS:file,nshown", end='')
        for method in sorted(rank[doc]):
            md = method[sortkeylen:]            
            print(",{}".format(md), end='')
        print()

    # compute and print number of keywords from method in GT 
    print("OUT:{},{}".format(doc,nshown), end='')
    foundKw[doc] = dict()
    for method in sorted(rank[doc]):
        foundKw[doc][method] = 0
        for rkval in range(1,maxRankShow+1):
            if rkval in rank[doc][method]:
                t = rank[doc][method][rkval][0]
                if t in list(GT[doc].values()):
                    foundKw[doc][method] += 1
        print(",{}".format(foundKw[doc][method]), end='')
    print()
    
## statistics
if not docMode:
    stats = dict()
    allmethods = []
    numberOfDocsByGTKw = dict()
    # loop over documents
    for fileCounter,docfn in enumerate(docList):
        if fileCounter >= testLimit:
            break
        doc = os.path.splitext(docfn)[0]
        if fileCounter == 0:
            # make the list of all methods
            allmethods = sorted(rank[doc])
        # number of keywords for this doc in ground truth
        gtkw = len(list(GT[doc].values()))
        if gtkw not in numberOfDocsByGTKw:
            numberOfDocsByGTKw[gtkw] = 1
        else:
            numberOfDocsByGTKw[gtkw] += 1
        if gtkw not in stats:
            # initialize stats dict
            stats[gtkw] = {method:dict() for method in allmethods}
        # loop over methods
        for method in allmethods:
            # success score = |ground truth keywords found by method in doc|
            fndkw = foundKw[doc][method]
            # count +1 for this doc, method, and success score value
            if fndkw not in stats[gtkw][method]:
                stats[gtkw][method][fndkw] = 1
            else:
                stats[gtkw][method][fndkw] += 1

    #pp.pprint(stats)
    print("|GT|,docs", end='')
    for method in allmethods:
        md = method[sortkeylen:]
        print(",{}".format(md), end='')
    print()
    for gtkw in sorted(list(stats)):
        if gtkw > 0:
            print("{},{},".format(gtkw, numberOfDocsByGTKw[gtkw]), end='')
            for cm,method in enumerate(allmethods):
                #print("[", end='')
                cf = 0
                for fndkw in stats[gtkw][method]:
                    if fndkw > 0:
                        if cf > 0:
                            print(" ", end='')                            
                        s = stats[gtkw][method][fndkw] 
                        print("{}@{}".format(s, fndkw), end='')
                        cf += 1
                #print("]", end='')
                if cm < len(allmethods)-1:
                    print(",", end='')
            print()
            

        
    
