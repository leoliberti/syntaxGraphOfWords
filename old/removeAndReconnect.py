# remove nodes with given prop, turn 2-paths through removed nodes with edges
# from <https://stackoverflow.com/questions/53353335/networkx-remove-node-and-reconnect-edges>
def removeAndReconnect(G:nx.Graph, node_removal_predicate:callable):
    '''
    Loop over the graph until all nodes that match the supplied predicate 
    have been removed and their incident edges fused.
    '''
    g = G.copy()
    while any(node_removal_predicate(node) for node in g.nodes):
        g0 = g.copy()
        for node in g.nodes:
            if node_removal_predicate(node):
                if g.is_directed():
                    in_edges_containing_node = list(g0.in_edges(node))
                    out_edges_containing_node = list(g0.out_edges(node))
                    for in_src,_ in in_edges_containing_node:
                        for _,out_dst in out_edges_containing_node:
                            #g0.add_edge(in_src, out_dst) # unweighted
                            dist = nx.shortest_path_length(g0, in_src, out_dst, weight='weight')
                            g0.add_edge(in_src, out_dst, weight=dist)
                else:
                    edges_containing_node = g.edges(node)
                    dst_to_link = [e[1] for e in edges_containing_node]
                    dst_pairs_to_link = list(combinations(dst_to_link, r = 2))
                    for pair in dst_pairs_to_link:
                        #g0.add_edge(pair[0], pair[1]) # unweighted
                        dist = nx.shortest_path_length(g0, pair[0], pair[1], weight='weight')
                        g0.add_edge(pair[0], pair[1], weight=dist)
                g0.remove_node(node)
                break
        g = g0
    return g
