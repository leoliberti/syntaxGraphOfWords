230301:
define a graph-of-words replacing the linear order of the words by syntax trees
choice between constituency, dependency, and sliding window with radius

Usage:
syntax is: ./graphwords2.py textfile [constituency|dependency|classic=radius]
  creates a graph-of-words from text in a textfile with various method
    graph-of-words vertices numbered from 1
  methods:
    constituency: word relations from constituency tree paths
    dependency:   word relations from dependency tree
    classic:      word relations from sliding window of given radius
    default is constituency

textfile should be a text file in English (best if ASCII-128 only)

The algorithm for classic Graphs-of-Words is: collect the list of tokens in the text, lose any token with POS not in ['VERB', 'NOUN', 'ADJ'] (this can be configured in GLOBALS, see importantPos variable), then from the resulting list take the sliding windows and add corresponding edges. Finally, contract token nodes having the same lemmatization (other contraction properties can be defined). Edge weights mean linear text distance (before contraction).

Algorithm for dependency-based Graphs-of-Words: find the dependency tree for each sentence in the text (using spacy), make a graph consisting of the disconnected union of these trees, contract nodes with same lemmatization (configurable), remove nodes with POS not in VERB,NOUN,ADJ (configurable) and reconnect nodes adjacent to removed ones with simple edges. Edge weights come from contractions and reconnections, and represent syntactic distance.

Algorithm for constituency-based Graphs-of-Words: like dependency (found with spacy and benepar), but there is a further operation: projection on the leaf nodes (the only nodes in constituency trees that contain tokens). Again, the edge weights come from contractions, reconnections, but also projection (every shortest path on the tree between unconnected token nodes becomes a simple edge with weight=shortest path length; shortest paths on trees are computed using lowest common ancestors). Edge weights represent syntactic distance.

Use python 3.10 (somewhere in the imports a module dependency requires it, I think).

---

Some instructions:
- unzip lit_corpus.zip into literature/ (it already contains pre-computed graphs-of-words in .dot files, so you can skip the next step)
- run "./gow_batch.sh literature/" to create graphs-of-words (4-proximity, constituency, dependency) on the whole literature/ corpus (this calls ./graphwords2.py multiple times)
- run "./tfidf.py literature/" to compute ranks on the whole literature/ corpus
