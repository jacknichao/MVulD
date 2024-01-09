import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
import re
import numpy as np
import pandas as pd
import sastvd as svd
import scipy.sparse as sparse
from graphviz import Digraph
from collections import Counter
import os
import pickle


def nodelabel2line(label: str):
    """Given a node label, return the line number.

    Example:
    s = "METHOD_1.0: static long main()..."
    nodelabel2line(s)
    >>> '1.0'
    """
    try:
        return str(int(label))
    except:
        return label.split(":")[0].split("_")[-1]


def randcolor():
    """Generate random color."""

    def r():
        return random.randint(0, 255)

    return "#%02X%02X%02X" % (r(), r(), r()) # change to HEX

type_map = {'UNKNOWN': 0, 'METHOD': 1, 'METHOD_PARAMETER_IN': 2, 'BLOCK': 3, 'External Function Call': 4,
            'Comparison Operator': 5, 'IDENTIFIER': 6, 'Assignment Operator': 7, 'RETURN': 8, 'LITERAL': 9,
            'METHOD_RETURN': 10, 'METHOD_PARAMETER_OUT': 11, 'IF': 12, 'Arithmetic Operator': 13,
            'Builtin Function Call': 14, 'Access Operator': 15, 'FIELD_IDENTIFIER': 16, 'Other Operator': 17,
            'LOCAL': 18, 'Logical Operator': 19, 'Cast Operator': 20, 'WHILE': 21, 'ELSE': 22, 'FOR': 23, 'GOTO': 24,
            'JUMP_TARGET': 25, 'SWITCH': 26, 'BREAK': 27, 'DO': 28, 'CONTINUE': 29, 'TYPE_DECL': 30, 'MEMBER': 31}
# Specified color
node_color_map = {'UNKNOWN': "#ffb3a7", 'METHOD': "#ed5736", 'METHOD_PARAMETER_IN': "#f00056", 'BLOCK': "#fff143", 
            'External Function Call': "#ffa631",'Comparison Operator': "#b25d25", 'IDENTIFIER': "#ccffff", 
            'Assignment Operator': "#bddd22", 'RETURN': "#B4EEB4", 'LITERAL': "#0eb83a",
            'METHOD_RETURN': "#1bd1a5", 'METHOD_PARAMETER_OUT': "#FFFACD", 'IF': "#44cef6", 'Arithmetic Operator': "#177cb0",
            'Builtin Function Call': "#B0C4DE", 'Access Operator': "#FFEBCD", 'FIELD_IDENTIFIER': "#8d4bbb", 'Other Operator': "#4b5cc4",
            'LOCAL': "#b0a4e3", 'Logical Operator': "#f2be45", 'Cast Operator': "#549688", 'WHILE': "#6b6882", 'ELSE': "#4c8dae", 
            'FOR':"#e4c6d0", 'GOTO': "#75878a",'JUMP_TARGET': "#a29b7c", 'SWITCH': "#25f8cb", 'BREAK': "#725e82", 
            'DO': "#E6E6FA", 'CONTINUE': "#CFCFCF", 'TYPE_DECL': "#FFE7BA", 'MEMBER': "#8B8B7A"}

def get_digraph_ast(nodes, edges, edges_ast,edge_label=False):
    ## based on ast structure,then adding other types of edges
    """Plote digraph given nodes and edges list.
    """
    dot = Digraph(comment="Combined PDG") 
    dot.attr('graph',size="1000",nodesep='0.5',ranksep='1.0',splines = "true") 
    # dot.attr('node', shape='box',width='0.8',height='0.5',fixedsize='true')
    

    nodes = [n + [nodelabel2line(n[1])] for n in nodes]
    
    colormap = {"": "white"}
    for n in nodes: 
        if n[3] not in colormap:
            # colormap[n[3]] = randcolor()
            colormap[n[3]] = node_color_map[n[3]]
            # print(colormap[n[3]]) 

    for n in nodes:
        str3 = str(n[1])
        # str3 = re.findall(r'.{25}|.+',str(n[1]))
        nodelable = '\n'.join(str3)
        # using label and id
        lable_and_id = str(n[1]).split(" ")[0] 

        lable_text = lable_and_id
        # lable_text = "" 
        if(str(n[0])=="1" or str(n[2])=="RETURN"): # Head and tail nodes
            style = {"style": "filled", "fillcolor": colormap[n[3]],"shape":"ellipse"}
            dot.node(str(n[0]), lable_text, **style) # str(n[1])
        else:
            style = {"style": "filled", "fillcolor": colormap[n[3]],"shape":"box"}  
            dot.node(str(n[0]), lable_text, **style) 
        # n[0]=linenumber n[1]=node_lable
    edgemap={}
    ## build ast
    for e in edges_ast:
        style = {"color": "black"} 
        style["color"] = "black" # darkgreen
        style["style"] = "bold"
        if(str(e[0])!=str(e[1])):
            value_list = []
            value_list.append(str(e[1])) 
            edgemap[str(e[0])] = value_list
            dot.edge(str(e[0]), str(e[1]), **style)
    for e in edges:
        style = {"color": "black"}
        if e[2] == "CALL":
            style["style"] = "solid"
            style["color"] = "purple"
        elif e[2] == "AST":
            style["style"] = "bold"
            style["color"] = "black" # darkgreen
            continue
            # style["constraint"] = "false" 
        elif e[2] == "CFG":
            style["style"] = "bold"
            style["color"] = "red"
            style["constraint"] = "false" 
        elif e[2] == "CDG":
            style["style"] = "bold"
            style["color"] = "blue"
            style["constraint"] = "false" 
        elif e[2] == "REACHING_DEF":
            style["style"] = "solid"
            style["color"] = "orange"
        elif "DDG" in e[2]:
            style["style"] = "dashed"
            style["color"] = "darkgreen"
        else:
            style["style"] = "solid"
            style["color"] = "black"
        style["penwidth"] = "1"
        if edge_label:
            # edges columns=["innode", "outnode", "etype", "dataflow"]
            dot.edge(str(e[0]), str(e[1]), e[2], **style)
        else:
            if(str(e[0])!=str(e[1])): # If it is self-ring, no edge is established
                ## Remove duplicate edges
                if str(e[0]) in edgemap:
                    if str(e[1]) in edgemap[str(e[0])]:
                        continue
                    else:
                        edgemap[str(e[0])].append(str(e[1]))
                        dot.edge(str(e[0]), str(e[1]), **style)
                else:
                    value_list = []
                    value_list.append(str(e[1])) 
                    edgemap[str(e[0])] = value_list
                    dot.edge(str(e[0]), str(e[1]), **style)
    # print(edgemap)   
    return dot

def get_words(id):
    submit_path = "./dict"
    
    submit_d1_file = os.path.join(submit_path, "d1.pkl")
    submit_d2_file = os.path.join(submit_path, "d2.pkl")
    submit_d1_file = os.path.abspath(submit_d1_file)
    submit_d2_file = os.path.abspath(submit_d2_file)

    with open(submit_d1_file, "rb") as tf:
        pd1 = pickle.load(tf)
    with open(submit_d2_file, "rb") as tf:
        pd2 = pickle.load(tf)
    words = pd1[id] # return map（id，word）
    return words  

def get_digraph(nodes, edges, edge_label=False):
    """Plote digraph given nodes and edges list.
    """
    dot = Digraph(comment="Combined PDG") 
    dot.attr('graph',size="1000",nodesep='0.5',ranksep='0.8',splines = "true")

    nodes = [n + [nodelabel2line(n[1])] for n in nodes]
    
    colormap = {"": "white"}
    for n in nodes: 
        if n[3] not in colormap:
            
            colormap[n[3]] = randcolor()
            # colormap[n[3]] = node_color_map[n[3]] # same ntype=same color
            # print(colormap[n[3]]) 
    
    for n in nodes:
        # 0
        str3 = str(n[1])
        # str3 = re.findall(r'.{25}|.+',str(n[1]))
        nodelable = '\n'.join(str3)
        lable_and_id = str(n[1]).split(" ")[0] 
        lable_text = str(n[2]).split("_")[0]+"+"+str(n[0]) 
        lable_text = str(n[1]) #
        # lable_text = str(n[0]) + str(n[1]).replace(lable_and_id, "")  
        # nodes[["id", "node_label","_label"]].to_numpy().tolist(),
        if(str(n[0])=="1" or str(n[2])=="RETURN"):
            # style = {"style": "filled", "fillcolor": colormap[str(n[0])],"shape":"ellipse"} # ellipse
            style = {"style": "solid", "shape":"ellipse"} #    
            dot.node(str(n[0]), lable_text, **style) # str(n[1])
        else:
            # style = {"style": "filled", "fillcolor": colormap[str(n[0])],"shape":"ellipse"} # before: box  
            style = {"style": "solid", "shape":"ellipse"}     
            dot.node(str(n[0]), lable_text, **style) 
    for e in edges:
        style = {"color": "black"}
        if e[2] == "CALL":
            style["style"] = "solid"
            style["color"] = "purple"
        elif e[2] == "AST":
            style["style"] = "bold"
            # style["style"] = "dashed"
            style["color"] = "black" # darkgreen
            # style["constraint"] = "false" 
        elif e[2] == "CFG":
            style["style"] = "bold"
            style["color"] = "red"
            # style["constraint"] = "false" # 
        elif e[2] == "CDG":
            style["style"] = "bold"
            style["color"] = "blue"
            # style["constraint"] = "false" 
        elif e[2] == "REACHING_DEF":
            style["style"] = "dashed" # dashed
            style["color"] = "blue"
        elif "DDG" in e[2]:
            style["style"] = "dashed"
            style["color"] = "darkgreen"
        else:
            style["style"] = "solid"
            style["color"] = "black"
        style["penwidth"] = "1"
        if edge_label:
            # edges columns=["innode", "outnode", "etype", "dataflow"]
            dot.edge(str(e[0]), str(e[1]), e[2], **style)
        else:
            if(str(e[0])!=str(e[1])): 
                dot.edge(str(e[0]), str(e[1]), **style)
    return dot


def run_joern(filepath: str, verbose: int):
    """Extract graph using most recent Joern."""
    script_file = svd.external_dir() / "get_func_graph.scala" 
    filename = svd.external_dir() / filepath #the position of .c document
    params = f"filename={filename}" 
    command = f"joern --script {script_file} --params='{params}'"
    command = str(svd.external_dir() / "joern-cli" / command) #run joern under corresponding path
    if verbose > 2:
        svd.debug(command)
    svd.subprocess_cmd(command, verbose=verbose)
    try:
        shutil.rmtree(svd.external_dir() / "joern-cli" / "workspace" / filename.name)
    except Exception as E:
        if verbose > 4:
            print(E)
        pass


def get_node_edges(filepath: str, verbose=0):
    """Get node and edges given filepath (must run after run_joern).

    filepath = "/home/david/Documents/projects/singularity-sastvd/storage/processed/bigvul/before/53.c"
    """
    outdir = Path(filepath).parent
    outfile = outdir / Path(filepath).name

    with open(str(outfile) + ".edges.json", "r") as f:
        edges = json.load(f) #return dict
        # print(edges)
        # innode = begin id，outnode= tail id
        edges = pd.DataFrame(edges, columns=["innode", "outnode", "etype", "dataflow"]) 
        edges = edges.fillna("") # replace NAN using ""
        # print(edges["etype"].value_counts()) # print etype of edges

    with open(str(outfile) + ".nodes.json", "r") as f:
        nodes = json.load(f)
        nodes = pd.DataFrame.from_records(nodes)
        if "controlStructureType" not in nodes.columns:
            nodes["controlStructureType"] = ""
        nodes = nodes.fillna("")

        try:
            nodes = nodes[
                ["id", "_label", "name", "code", "lineNumber", "controlStructureType"]
            ]
            nodes = nodes[(nodes['name'] != '<global>')]
            nodes = nodes[~nodes['_label'].apply(lambda x: 'META' in x)]
            # print(nodes.head(30))
        except Exception as E:
            if verbose > 1:
                svd.debug(f"Failed {filepath}: {E}")
            return None

    # Assign line number to local variables
    # with open(filepath, "r") as f:
    #     code = f.readlines()
    # lmap = assign_line_num_to_local(nodes, edges, code) 
    #     # old joern version 
    # nodes.lineNumber = nodes.apply(
    #     lambda x: lmap[x.id] if x.id in lmap else x.lineNumber, axis=1
    # )
    
    nodes = nodes.fillna("") 

    # Assign node name to node code if code is null
    nodes.code = nodes.apply(lambda x: "" if x.code == "<empty>" else x.code, axis=1)
    nodes.code = nodes.apply(lambda x: x.code if x.code != "" else x["name"], axis=1)
    # Assign node label for printing in the graph
    nodes["node_label"] = (
        nodes._label + "_" + nodes.lineNumber.astype(str) + ": " + nodes.code
    )

    # Filter by node type
    nodes = nodes[nodes._label != "COMMENT"] 
    nodes = nodes[nodes._label != "FILE"]

    # Filter by edge type
    edges = edges[edges.etype != "CONTAINS"]
    edges = edges[edges.etype != "SOURCE_FILE"]
    edges = edges[edges.etype != "DOMINATE"]
    edges = edges[edges.etype != "POST_DOMINATE"]

    # Remove nodes not connected to line number nodes (maybe not efficient)
    edges = edges.merge(
        nodes[["id", "lineNumber"]].rename(columns={"lineNumber": "line_out"}),
        left_on="outnode",
        right_on="id",
    )
    edges = edges.merge(
        nodes[["id", "lineNumber"]].rename(columns={"lineNumber": "line_in"}),
        left_on="innode",
        right_on="id",
    )
    edges = edges[(edges.line_out != "") | (edges.line_in != "")]

    # Uniquify types
    edges.outnode = edges.apply(
        lambda x: f"{x.outnode}_{x.innode}" if x.line_out == "" else x.outnode, axis=1
    )

    
    typemap = nodes[["id", "name"]].set_index("id").to_dict()["name"]
    linemap = nodes.set_index("id").to_dict()["lineNumber"]
    for e in edges.itertuples():
        if type(e.outnode) == str:
            lineNum = linemap[e.innode]
            node_label = f"TYPE_{lineNum}: {typemap[int(e.outnode.split('_')[0])]}"
            # Use pandas.concat instead：原先append
            nodes = nodes.append(
                {"id": e.outnode, "node_label": node_label, "lineNumber": lineNum},
                ignore_index=True,
            )
            
            # nodes = pd.concat(
            #     [nodes,{"id": e.outnode, "node_label": node_label, "lineNumber": lineNum}],
            #     join='outer', 
            #     ignore_index = True,
            #     axis = 0  
            # )

    return nodes, edges


def plot_node_edges(filepath: str, lineNumber: int = -1, filter_edges=[]):
    """Plot node edges given filepath (must run after get_node_edges).

    TO BE DEPRECATED.
    """
    nodes, edges = get_node_edges(filepath)

    if len(filter_edges) > 0:
        edges = edges[edges.etype.isin(filter_edges)]

    # Draw graph
    if lineNumber > 0:
        nodesforline = set(nodes[nodes.lineNumber == lineNumber].id.tolist())
    else:
        nodesforline = set(nodes.id.tolist())

    edges_new = edges[
        (edges.outnode.isin(nodesforline)) | (edges.innode.isin(nodesforline))
    ]
    nodes_new = nodes[
        nodes.id.isin(set(edges_new.outnode.tolist() + edges_new.innode.tolist()))
    ]
    
    dot = get_digraph(
        nodes_new[["id", "node_label"]].to_numpy().tolist(),
        edges_new[["outnode", "innode", "etype"]].to_numpy().tolist(),
    )
    
    dot.render("/tmp/tmp.gv", view=True)


def full_run_joern(filepath: str, verbose=0): 
    """Run full Joern extraction and save output."""
    try:
        run_joern(filepath, verbose) #run joern to generate node.json and edge.json
        nodes, edges = get_node_edges(filepath) # transform .json to df
        return {"nodes": nodes, "edges": edges}
    except Exception as E:
        if verbose > 0:
            svd.debug(f"Failed {filepath}: {E}")
        return None


def full_run_joern_from_string(code: str, dataset: str, iid: str, verbose=0):
    """Run full joern from a string instead of file."""
    savedir = svd.get_dir(svd.interim_dir() / dataset)
    savepath = savedir / f"{iid}.c"
    with open(savepath, "w") as f:
        f.write(code)
    return full_run_joern(savepath, verbose)


def neighbour_nodes(nodes, edges, nodeids: list, hop: int = 1, intermediate=True):
    """Given nodes, edges, nodeid, return hop neighbours.
    nodes = pd.DataFrame()

    """
    nodes_new = (
        nodes.reset_index(drop=True).reset_index().rename(columns={"index": "adj"})
    )
    id2adj = pd.Series(nodes_new.adj.values, index=nodes_new.id).to_dict()
    adj2id = {v: k for k, v in id2adj.items()}

    arr = []
    for e in zip(edges.innode.map(id2adj), edges.outnode.map(id2adj)):
        arr.append([e[0], e[1]])
        arr.append([e[1], e[0]])

    arr = np.array(arr)
    shape = tuple(arr.max(axis=0)[:2] + 1)
    coo = sparse.coo_matrix((np.ones(len(arr)), (arr[:, 0], arr[:, 1])), shape=shape)

    def nodeid_neighbours_from_csr(nodeid):
        return [
            adj2id[i]
            for i in csr[
                id2adj[nodeid],
            ]
            .toarray()[0]
            .nonzero()[0]
        ]

    neighbours = defaultdict(list)
    if intermediate:
        for h in range(1, hop + 1):
            csr = coo.tocsr()
            csr **= h
            for nodeid in nodeids:
                neighbours[nodeid] += nodeid_neighbours_from_csr(nodeid)
        return neighbours
    else:
        csr = coo.tocsr()
        csr **= hop
        for nodeid in nodeids:
            neighbours[nodeid] += nodeid_neighbours_from_csr(nodeid)
        return neighbours


def rdg(edges, gtype):
    """Reduce graph given type.:AST, CFG,CDG, and DDG"""
    # print("rdg")
    if gtype == "reftype":
        return edges[(edges.etype == "EVAL_TYPE") | (edges.etype == "REF")]
    if gtype == "ast":
        return edges[(edges.etype == "AST")]
    if gtype == "pdg":
        return edges[(edges.etype == "REACHING_DEF") | (edges.etype == "CDG")]
    if gtype == "cfg":
        return edges[(edges.etype == "CFG")]
    if gtype == "cdg":
        return edges[(edges.etype == "CDG")]
    if gtype == "cfgcdg":
        return edges[(edges.etype == "CFG") | (edges.etype == "CDG")]
    if gtype == "all":
        return edges[
            (edges.etype == "CFG") 
            |(edges.etype == "CDG") 
            |(edges.etype == "AST")
            # | (edges.etype == "EVAL_TYPE")
            # | (edges.etype == "REF")
            # |(edges.etype == "REACHING_DEF")    
        ]
    if gtype == "other":
        return edges[
            (edges.etype == "CFG") 
            |(edges.etype == "CDG") #
            # |(edges.etype == "AST")
            # | (edges.etype == "EVAL_TYPE")
            # | (edges.etype == "REF")
            |(edges.etype == "REACHING_DEF")    
        ]


def assign_line_num_to_local(nodes, edges, code): 
    """Assign line number to local variable in CPG."""
    
    label_nodes = nodes[nodes._label == "LOCAL"].id.tolist()
    onehop_labels = neighbour_nodes(nodes, rdg(edges, "ast"), label_nodes, 1, False)
    twohop_labels = neighbour_nodes(nodes, rdg(edges, "reftype"), label_nodes, 2, False)
    node_types = nodes[nodes._label == "TYPE"]
    id2name = pd.Series(node_types.name.values, index=node_types.id).to_dict()
    node_blocks = nodes[
        (nodes._label == "BLOCK") | (nodes._label == "CONTROL_STRUCTURE")
    ]
    blocknode2line = pd.Series(
        node_blocks.lineNumber.values, index=node_blocks.id
    ).to_dict()
    local_vars = dict()
    local_vars_block = dict()
    for k, v in twohop_labels.items():
        types = [i for i in v if i in id2name and i < 1000]
        if len(types) == 0:
            continue
        assert len(types) == 1, "Incorrect Type Assumption."
        block = onehop_labels[k]
        assert len(block) == 1, "Incorrect block Assumption."
        block = block[0]
        local_vars[k] = id2name[types[0]]
        local_vars_block[k] = blocknode2line[block]
    nodes["local_type"] = nodes.id.map(local_vars)
    nodes["local_block"] = nodes.id.map(local_vars_block)
    local_line_map = dict()
    # print("---------------assign_line_num_to_local-------------------1")
    for row in nodes.dropna().itertuples():
        # print("---------------assign_line_num_to_local-------------------2")
        # print(["".join(i.split()) for i in code][int(row.local_block) :])
        localstr = "".join((row.local_type + row.name).split()) + ";"
        try:
            ln = ["".join(i.split()) for i in code][int(row.local_block) :].index(
                localstr
            )
            rel_ln = row.local_block + ln + 1
            local_line_map[row.id] = rel_ln
        except:
            continue
    return local_line_map


def drop_lone_nodes(nodes, edges):
    """Remove nodes with no edge connections.

    Args:
        nodes (pd.DataFrame): columns are id, node_label
        edges (pd.DataFrame): columns are outnode, innode, etype
    """
    nodes = nodes[(nodes.id.isin(edges.innode)) | (nodes.id.isin(edges.outnode))]
    return nodes


def plot_graph_node_edge_df(
    nodes, edges, nodeids=[], hop=1, drop_lone_nodes=True, edge_label=False
):
    """Plot graph from node and edge dataframes.

    Args:
        nodes (pd.DataFrame): columns are id, node_label
        edges (pd.DataFrame): columns are outnode, innode, etype
        drop_lone_nodes (bool): hide nodes with no in/out edges.
        lineNumber (int): Plot subgraph around this node.
    """
    # Drop lone nodes:Remove nodes with no edge connections
    if drop_lone_nodes:
        nodes = nodes[(nodes.id.isin(edges.innode)) | (nodes.id.isin(edges.outnode))]

    # Get subgraph
    if len(nodeids) > 0:
        nodeids = nodes[nodes.lineNumber.isin(nodeids)].id
        keep_nodes = neighbour_nodes(nodes, edges, nodeids, hop)
        keep_nodes = set(list(nodeids) + [i for j in keep_nodes.values() for i in j])
        nodes = nodes[nodes.id.isin(keep_nodes)]
        edges = edges[
            (edges.innode.isin(keep_nodes)) & (edges.outnode.isin(keep_nodes))
        ]

    # print("--------------plot_graph_node_edge_df-------------")
    dot = get_digraph(
        # nodes[["id", "node_label","_label","ntype"]].to_numpy().tolist(),
        nodes[["id", "node_label","_label"]].to_numpy().tolist(),
        edges[["outnode", "innode", "etype"]].to_numpy().tolist(),
        edge_label=edge_label, # false = don't show etype 
    )
    
    # dot.render("/tmp/tmp.gv",view=True) 
    return dot

def plot_graph_node_edge_df_basedonAST(
    nodes, edges,edges_ast, drop_lone_nodes=True, edge_label=False
):
    """Plot graph from node and edge dataframes.

    Args:
        nodes (pd.DataFrame): columns are id, node_label
        edges (pd.DataFrame): columns are outnode, innode, etype
        drop_lone_nodes (bool): hide nodes with no in/out edges.
        lineNumber (int): Plot subgraph around this node.
    """
    # Drop lone nodes:Remove nodes with no edge connections
    if drop_lone_nodes:
        nodes = nodes[(nodes.id.isin(edges.innode)) | (nodes.id.isin(edges.outnode))]

    dot = get_digraph_ast(
        nodes[["id", "node_label","_label","ntype"]].to_numpy().tolist(),
        edges[["outnode", "innode", "etype"]].to_numpy().tolist(),
        edges_ast [["outnode", "innode", "etype"]].to_numpy().tolist(),
        edge_label=edge_label, 
    )
    return dot

def type_2_type(info):
    for i in range(1):
        if info['_label'] == "CALL":
            if "<operator>" in info["name"]:
                if "assignment" in info["name"]:
                    new_type = "Assignment Operator"
                    continue
                if (
                        "addition" in info["name"]
                        or "subtraction" in info["name"]
                        or "division" in info["name"]
                        or "Plus" in info["name"]
                        or "Minus" in info["name"]
                        or "minus" in info["name"]
                        or "plus" in info["name"]
                        or "modulo" in info["name"]
                        or "multiplication" in info["name"]
                ):
                    new_type = "Arithmetic Operator"
                    continue
                if (
                        "lessThan" in info["name"]
                        or "greaterThan" in info["name"]
                        or "EqualsThan" in info["name"]
                        or "equals" in info["name"]
                ):
                    new_type = "Comparison Operator"
                    continue
                if (
                        "FieldAccess" in info["name"]
                        or "IndexAccess" in info["name"]
                        or "fieldAccess" in info["name"]
                        or "indexAccess" in info["name"]
                ):
                    new_type = "Access Operator"
                    continue
                if (
                        "logical" in info["name"]
                        or "<operator>.not" in info["name"]
                        or "<operator>.or" in info["name"]
                        or "<operator>.and" in info["name"]
                        or "conditional" in info["name"]
                ):
                    new_type = "Logical Operator"
                    continue
                if "<operator>.cast" in info["name"]:
                    new_type = "Cast Operator"
                    continue
                if "<operator>" in info["name"]:
                    new_type = "Other Operator"
                    continue
            elif info["name"] in l_funcs:
                new_type = "Builtin Function Call"
                continue
            else:
                new_type = "External Function Call"
                continue
        if info["_label"] == "CONTROL_STRUCTURE":
            new_type = info["controlStructureType"]
            continue
        new_type = info["_label"]
    return new_type
    
def count_labels(nodes):
    """Get info about nodes."""
    label_info = []
    for _, info in nodes.iterrows():
        new_type = type_2_type(info)
        label_info.append(new_type)

    counter = Counter(label_info)
    return counter, label_info

l_funcs = set([
    "StrNCat",
    "getaddrinfo",
    "_ui64toa",
    "fclose",
    "pthread_mutex_lock",
    "gets_s",
    "sleep",
    "_ui64tot",
    "freopen_s",
    "_ui64tow",
    "send",
    "lstrcat",
    "HMAC_Update",
    "__fxstat",
    "StrCatBuff",
    "_mbscat",
    "_mbstok_s",
    "_cprintf_s",
    "ldap_search_init_page",
    "memmove_s",
    "ctime_s",
    "vswprintf",
    "vswprintf_s",
    "_snwprintf",
    "_gmtime_s",
    "_tccpy",
    "*RC6*",
    "_mbslwr_s",
    "random",
    "__wcstof_internal",
    "_wcslwr_s",
    "_ctime32_s",
    "wcsncat*",
    "MD5_Init",
    "_ultoa",
    "snprintf",
    "memset",
    "syslog",
    "_vsnprintf_s",
    "HeapAlloc",
    "pthread_mutex_destroy",
    "ChangeWindowMessageFilter",
    "_ultot",
    "crypt_r",
    "_strupr_s_l",
    "LoadLibraryExA",
    "_strerror_s",
    "LoadLibraryExW",
    "wvsprintf",
    "MoveFileEx",
    "_strdate_s",
    "SHA1",
    "sprintfW",
    "StrCatNW",
    "_scanf_s_l",
    "pthread_attr_init",
    "_wtmpnam_s",
    "snscanf",
    "_sprintf_s_l",
    "dlopen",
    "sprintfA",
    "timed_mutex",
    "OemToCharA",
    "ldap_delete_ext",
    "sethostid",
    "popen",
    "OemToCharW",
    "_gettws",
    "vfork",
    "_wcsnset_s_l",
    "sendmsg",
    "_mbsncat",
    "wvnsprintfA",
    "HeapFree",
    "_wcserror_s",
    "realloc",
    "_snprintf*",
    "wcstok",
    "_strncat*",
    "StrNCpy",
    "_wasctime_s",
    "push*",
    "_lfind_s",
    "CC_SHA512",
    "ldap_compare_ext_s",
    "wcscat_s",
    "strdup",
    "_chsize_s",
    "sprintf_s",
    "CC_MD4_Init",
    "wcsncpy",
    "_wfreopen_s",
    "_wcsupr_s",
    "_searchenv_s",
    "ldap_modify_ext_s",
    "_wsplitpath",
    "CC_SHA384_Final",
    "MD2",
    "RtlCopyMemory",
    "lstrcatW",
    "MD4",
    "MD5",
    "_wcstok_s_l",
    "_vsnwprintf_s",
    "ldap_modify_s",
    "strerror",
    "_lsearch_s",
    "_mbsnbcat_s",
    "_wsplitpath_s",
    "MD4_Update",
    "_mbccpy_s",
    "_strncpy_s_l",
    "_snprintf_s",
    "CC_SHA512_Init",
    "fwscanf_s",
    "_snwprintf_s",
    "CC_SHA1",
    "swprintf",
    "fprintf",
    "EVP_DigestInit_ex",
    "strlen",
    "SHA1_Init",
    "strncat",
    "_getws_s",
    "CC_MD4_Final",
    "wnsprintfW",
    "lcong48",
    "lrand48",
    "write",
    "HMAC_Init",
    "_wfopen_s",
    "wmemchr",
    "_tmakepath",
    "wnsprintfA",
    "lstrcpynW",
    "scanf_s",
    "_mbsncpy_s_l",
    "_localtime64_s",
    "fstream.open",
    "_wmakepath",
    "Connection.open",
    "_tccat",
    "valloc",
    "setgroups",
    "unlink",
    "fstream.put",
    "wsprintfA",
    "*SHA1*",
    "_wsearchenv_s",
    "ualstrcpyA",
    "CC_MD5_Update",
    "strerror_s",
    "HeapCreate",
    "ualstrcpyW",
    "__xstat",
    "_wmktemp_s",
    "StrCatChainW",
    "ldap_search_st",
    "_mbstowcs_s_l",
    "ldap_modify_ext",
    "_mbsset_s",
    "strncpy_s",
    "move",
    "execle",
    "StrCat",
    "xrealloc",
    "wcsncpy_s",
    "_tcsncpy*",
    "execlp",
    "RIPEMD160_Final",
    "ldap_search_s",
    "EnterCriticalSection",
    "_wctomb_s_l",
    "fwrite",
    "_gmtime64_s",
    "sscanf_s",
    "wcscat",
    "_strupr_s",
    "wcrtomb_s",
    "VirtualLock",
    "ldap_add_ext_s",
    "_mbscpy",
    "_localtime32_s",
    "lstrcpy",
    "_wcsncpy*",
    "CC_SHA1_Init",
    "_getts",
    "_wfopen",
    "__xstat64",
    "strcoll",
    "_fwscanf_s_l",
    "_mbslwr_s_l",
    "RegOpenKey",
    "makepath",
    "seed48",
    "CC_SHA256",
    "sendto",
    "execv",
    "CalculateDigest",
    "memchr",
    "_mbscpy_s",
    "_strtime_s",
    "ldap_search_ext_s",
    "_chmod",
    "flock",
    "__fxstat64",
    "_vsntprintf",
    "CC_SHA256_Init",
    "_itoa_s",
    "__wcserror_s",
    "_gcvt_s",
    "fstream.write",
    "sprintf",
    "recursive_mutex",
    "strrchr",
    "gethostbyaddr",
    "_wcsupr_s_l",
    "strcspn",
    "MD5_Final",
    "asprintf",
    "_wcstombs_s_l",
    "_tcstok",
    "free",
    "MD2_Final",
    "asctime_s",
    "_alloca",
    "_wputenv_s",
    "_wcsset_s",
    "_wcslwr_s_l",
    "SHA1_Update",
    "filebuf.sputc",
    "filebuf.sputn",
    "SQLConnect",
    "ldap_compare",
    "mbstowcs_s",
    "HMAC_Final",
    "pthread_condattr_init",
    "_ultow_s",
    "rand",
    "ofstream.put",
    "CC_SHA224_Final",
    "lstrcpynA",
    "bcopy",
    "system",
    "CreateFile*",
    "wcscpy_s",
    "_mbsnbcpy*",
    "open",
    "_vsnwprintf",
    "strncpy",
    "getopt_long",
    "CC_SHA512_Final",
    "_vsprintf_s_l",
    "scanf",
    "mkdir",
    "_localtime_s",
    "_snprintf",
    "_mbccpy_s_l",
    "memcmp",
    "final",
    "_ultoa_s",
    "lstrcpyW",
    "LoadModule",
    "_swprintf_s_l",
    "MD5_Update",
    "_mbsnset_s_l",
    "_wstrtime_s",
    "_strnset_s",
    "lstrcpyA",
    "_mbsnbcpy_s",
    "mlock",
    "IsBadHugeWritePtr",
    "copy",
    "_mbsnbcpy_s_l",
    "wnsprintf",
    "wcscpy",
    "ShellExecute",
    "CC_MD4",
    "_ultow",
    "_vsnwprintf_s_l",
    "lstrcpyn",
    "CC_SHA1_Final",
    "vsnprintf",
    "_mbsnbset_s",
    "_i64tow",
    "SHA256_Init",
    "wvnsprintf",
    "RegCreateKey",
    "strtok_s",
    "_wctime32_s",
    "_i64toa",
    "CC_MD5_Final",
    "wmemcpy",
    "WinExec",
    "CreateDirectory*",
    "CC_SHA256_Update",
    "_vsnprintf_s_l",
    "jrand48",
    "wsprintf",
    "ldap_rename_ext_s",
    "filebuf.open",
    "_wsystem",
    "SHA256_Update",
    "_cwscanf_s",
    "wsprintfW",
    "_sntscanf",
    "_splitpath",
    "fscanf_s",
    "strpbrk",
    "wcstombs_s",
    "wscanf",
    "_mbsnbcat_s_l",
    "strcpynA",
    "pthread_cond_init",
    "wcsrtombs_s",
    "_wsopen_s",
    "CharToOemBuffA",
    "RIPEMD160_Update",
    "_tscanf",
    "HMAC",
    "StrCCpy",
    "Connection.connect",
    "lstrcatn",
    "_mbstok",
    "_mbsncpy",
    "CC_SHA384_Update",
    "create_directories",
    "pthread_mutex_unlock",
    "CFile.Open",
    "connect",
    "_vswprintf_s_l",
    "_snscanf_s_l",
    "fputc",
    "_wscanf_s",
    "_snprintf_s_l",
    "strtok",
    "_strtok_s_l",
    "lstrcatA",
    "snwscanf",
    "pthread_mutex_init",
    "fputs",
    "CC_SHA384_Init",
    "_putenv_s",
    "CharToOemBuffW",
    "pthread_mutex_trylock",
    "__wcstoul_internal",
    "_memccpy",
    "_snwprintf_s_l",
    "_strncpy*",
    "wmemset",
    "MD4_Init",
    "*RC4*",
    "strcpyW",
    "_ecvt_s",
    "memcpy_s",
    "erand48",
    "IsBadHugeReadPtr",
    "strcpyA",
    "HeapReAlloc",
    "memcpy",
    "ldap_rename_ext",
    "fopen_s",
    "srandom",
    "_cgetws_s",
    "_makepath",
    "SHA256_Final",
    "remove",
    "_mbsupr_s",
    "pthread_mutexattr_init",
    "__wcstold_internal",
    "StrCpy",
    "ldap_delete",
    "wmemmove_s",
    "_mkdir",
    "strcat",
    "_cscanf_s_l",
    "StrCAdd",
    "swprintf_s",
    "_strnset_s_l",
    "close",
    "ldap_delete_ext_s",
    "ldap_modrdn",
    "strchr",
    "_gmtime32_s",
    "_ftcscat",
    "lstrcatnA",
    "_tcsncat",
    "OemToChar",
    "mutex",
    "CharToOem",
    "strcpy_s",
    "lstrcatnW",
    "_wscanf_s_l",
    "__lxstat64",
    "memalign",
    "MD2_Init",
    "StrCatBuffW",
    "StrCpyN",
    "CC_MD5",
    "StrCpyA",
    "StrCatBuffA",
    "StrCpyW",
    "tmpnam_r",
    "_vsnprintf",
    "strcatA",
    "StrCpyNW",
    "_mbsnbset_s_l",
    "EVP_DigestInit",
    "_stscanf",
    "CC_MD2",
    "_tcscat",
    "StrCpyNA",
    "xmalloc",
    "_tcslen",
    "*MD4*",
    "vasprintf",
    "strxfrm",
    "chmod",
    "ldap_add_ext",
    "alloca",
    "_snscanf_s",
    "IsBadWritePtr",
    "swscanf_s",
    "wmemcpy_s",
    "_itoa",
    "_ui64toa_s",
    "EVP_DigestUpdate",
    "__wcstol_internal",
    "_itow",
    "StrNCatW",
    "strncat_s",
    "ualstrcpy",
    "execvp",
    "_mbccat",
    "EVP_MD_CTX_init",
    "assert",
    "ofstream.write",
    "ldap_add",
    "_sscanf_s_l",
    "drand48",
    "CharToOemW",
    "swscanf",
    "_itow_s",
    "RIPEMD160_Init",
    "CopyMemory",
    "initstate",
    "getpwuid",
    "vsprintf",
    "_fcvt_s",
    "CharToOemA",
    "setuid",
    "malloc",
    "StrCatNA",
    "strcat_s",
    "srand",
    "getwd",
    "_controlfp_s",
    "olestrcpy",
    "__wcstod_internal",
    "_mbsnbcat",
    "lstrncat",
    "des_*",
    "CC_SHA224_Init",
    "set*",
    "vsprintf_s",
    "SHA1_Final",
    "_umask_s",
    "gets",
    "setstate",
    "wvsprintfW",
    "LoadLibraryEx",
    "ofstream.open",
    "calloc",
    "_mbstrlen",
    "_cgets_s",
    "_sopen_s",
    "IsBadStringPtr",
    "wcsncat_s",
    "add*",
    "nrand48",
    "create_directory",
    "ldap_search_ext",
    "_i64toa_s",
    "_ltoa_s",
    "_cwscanf_s_l",
    "wmemcmp",
    "__lxstat",
    "lstrlen",
    "pthread_condattr_destroy",
    "_ftcscpy",
    "wcstok_s",
    "__xmknod",
    "pthread_attr_destroy",
    "sethostname",
    "_fscanf_s_l",
    "StrCatN",
    "RegEnumKey",
    "_tcsncpy",
    "strcatW",
    "AfxLoadLibrary",
    "setenv",
    "tmpnam",
    "_mbsncat_s_l",
    "_wstrdate_s",
    "_wctime64_s",
    "_i64tow_s",
    "CC_MD4_Update",
    "ldap_add_s",
    "_umask",
    "CC_SHA1_Update",
    "_wcsset_s_l",
    "_mbsupr_s_l",
    "strstr",
    "_tsplitpath",
    "memmove",
    "_tcscpy",
    "vsnprintf_s",
    "strcmp",
    "wvnsprintfW",
    "tmpfile",
    "ldap_modify",
    "_mbsncat*",
    "mrand48",
    "sizeof",
    "StrCatA",
    "_ltow_s",
    "*desencrypt*",
    "StrCatW",
    "_mbccpy",
    "CC_MD2_Init",
    "RIPEMD160",
    "ldap_search",
    "CC_SHA224",
    "mbsrtowcs_s",
    "update",
    "ldap_delete_s",
    "getnameinfo",
    "*RC5*",
    "_wcsncat_s_l",
    "DriverManager.getConnection",
    "socket",
    "_cscanf_s",
    "ldap_modrdn_s",
    "_wopen",
    "CC_SHA256_Final",
    "_snwprintf*",
    "MD2_Update",
    "strcpy",
    "_strncat_s_l",
    "CC_MD5_Init",
    "mbscpy",
    "wmemmove",
    "LoadLibraryW",
    "_mbslen",
    "*alloc",
    "_mbsncat_s",
    "LoadLibraryA",
    "fopen",
    "StrLen",
    "delete",
    "_splitpath_s",
    "CreateFileTransacted*",
    "MD4_Final",
    "_open",
    "CC_SHA384",
    "wcslen",
    "wcsncat",
    "_mktemp_s",
    "pthread_mutexattr_destroy",
    "_snwscanf_s",
    "_strset_s",
    "_wcsncpy_s_l",
    "CC_MD2_Final",
    "_mbstok_s_l",
    "wctomb_s",
    "MySQL_Driver.connect",
    "_snwscanf_s_l",
    "*_des_*",
    "LoadLibrary",
    "_swscanf_s_l",
    "ldap_compare_s",
    "ldap_compare_ext",
    "_strlwr_s",
    "GetEnvironmentVariable",
    "cuserid",
    "_mbscat_s",
    "strspn",
    "_mbsncpy_s",
    "ldap_modrdn2",
    "LeaveCriticalSection",
    "CopyFile",
    "getpwd",
    "sscanf",
    "creat",
    "RegSetValue",
    "ldap_modrdn2_s",
    "CFile.Close",
    "*SHA_1*",
    "pthread_cond_destroy",
    "CC_SHA512_Update",
    "*RC2*",
    "StrNCatA",
    "_mbsnbcpy",
    "_mbsnset_s",
    "crypt",
    "excel",
    "_vstprintf",
    "xstrdup",
    "wvsprintfA",
    "getopt",
    "mkstemp",
    "_wcsnset_s",
    "_stprintf",
    "_sntprintf",
    "tmpfile_s",
    "OpenDocumentFile",
    "_mbsset_s_l",
    "_strset_s_l",
    "_strlwr_s_l",
    "ifstream.open",
    "xcalloc",
    "StrNCpyA",
    "_wctime_s",
    "CC_SHA224_Update",
    "_ctime64_s",
    "MoveFile",
    "chown",
    "StrNCpyW",
    "IsBadReadPtr",
    "_ui64tow_s",
    "IsBadCodePtr",
    "getc",
    "OracleCommand.ExecuteOracleScalar",
    "AccessDataSource.Insert",
    "IDbDataAdapter.FillSchema",
    "IDbDataAdapter.Update",
    "GetWindowText*",
    "SendMessage",
    "SqlCommand.ExecuteNonQuery",
    "streambuf.sgetc",
    "streambuf.sgetn",
    "OracleCommand.ExecuteScalar",
    "SqlDataSource.Update",
    "_Read_s",
    "IDataAdapter.Fill",
    "_wgetenv",
    "_RecordsetPtr.Open*",
    "AccessDataSource.Delete",
    "Recordset.Open*",
    "filebuf.sbumpc",
    "DDX_*",
    "RegGetValue",
    "fstream.read*",
    "SqlCeCommand.ExecuteResultSet",
    "SqlCommand.ExecuteXmlReader",
    "main",
    "streambuf.sputbackc",
    "read",
    "m_lpCmdLine",
    "CRichEditCtrl.Get*",
    "istream.putback",
    "SqlCeCommand.ExecuteXmlReader",
    "SqlCeCommand.BeginExecuteXmlReader",
    "filebuf.sgetn",
    "OdbcDataAdapter.Update",
    "filebuf.sgetc",
    "SQLPutData",
    "recvfrom",
    "OleDbDataAdapter.FillSchema",
    "IDataAdapter.FillSchema",
    "CRichEditCtrl.GetLine",
    "DbDataAdapter.Update",
    "SqlCommand.ExecuteReader",
    "istream.get",
    "ReceiveFrom",
    "_main",
    "fgetc",
    "DbDataAdapter.FillSchema",
    "kbhit",
    "UpdateCommand.Execute*",
    "Statement.execute",
    "fgets",
    "SelectCommand.Execute*",
    "getch",
    "OdbcCommand.ExecuteNonQuery",
    "CDaoQueryDef.Execute",
    "fstream.getline",
    "ifstream.getline",
    "SqlDataAdapter.FillSchema",
    "OleDbCommand.ExecuteReader",
    "Statement.execute*",
    "SqlCeCommand.BeginExecuteNonQuery",
    "OdbcCommand.ExecuteScalar",
    "SqlCeDataAdapter.Update",
    "sendmessage",
    "mysqlpp.DBDriver",
    "fstream.peek",
    "Receive",
    "CDaoRecordset.Open",
    "OdbcDataAdapter.FillSchema",
    "_wgetenv_s",
    "OleDbDataAdapter.Update",
    "readsome",
    "SqlCommand.BeginExecuteXmlReader",
    "recv",
    "ifstream.peek",
    "_Main",
    "_tmain",
    "_Readsome_s",
    "SqlCeCommand.ExecuteReader",
    "OleDbCommand.ExecuteNonQuery",
    "fstream.get",
    "IDbCommand.ExecuteScalar",
    "filebuf.sputbackc",
    "IDataAdapter.Update",
    "streambuf.sbumpc",
    "InsertCommand.Execute*",
    "RegQueryValue",
    "IDbCommand.ExecuteReader",
    "SqlPipe.ExecuteAndSend",
    "Connection.Execute*",
    "getdlgtext",
    "ReceiveFromEx",
    "SqlDataAdapter.Update",
    "RegQueryValueEx",
    "SQLExecute",
    "pread",
    "SqlCommand.BeginExecuteReader",
    "AfxWinMain",
    "getchar",
    "istream.getline",
    "SqlCeDataAdapter.Fill",
    "OleDbDataReader.ExecuteReader",
    "SqlDataSource.Insert",
    "istream.peek",
    "SendMessageCallback",
    "ifstream.read*",
    "SqlDataSource.Select",
    "SqlCommand.ExecuteScalar",
    "SqlDataAdapter.Fill",
    "SqlCommand.BeginExecuteNonQuery",
    "getche",
    "SqlCeCommand.BeginExecuteReader",
    "getenv",
    "streambuf.snextc",
    "Command.Execute*",
    "_CommandPtr.Execute*",
    "SendNotifyMessage",
    "OdbcDataAdapter.Fill",
    "AccessDataSource.Update",
    "fscanf",
    "QSqlQuery.execBatch",
    "DbDataAdapter.Fill",
    "cin",
    "DeleteCommand.Execute*",
    "QSqlQuery.exec",
    "PostMessage",
    "ifstream.get",
    "filebuf.snextc",
    "IDbCommand.ExecuteNonQuery",
    "Winmain",
    "fread",
    "getpass",
    "GetDlgItemTextCCheckListBox.GetCheck",
    "DISP_PROPERTY_EX",
    "pread64",
    "Socket.Receive*",
    "SACommand.Execute*",
    "SQLExecDirect",
    "SqlCeDataAdapter.FillSchema",
    "DISP_FUNCTION",
    "OracleCommand.ExecuteNonQuery",
    "CEdit.GetLine",
    "OdbcCommand.ExecuteReader",
    "CEdit.Get*",
    "AccessDataSource.Select",
    "OracleCommand.ExecuteReader",
    "OCIStmtExecute",
    "getenv_s",
    "DB2Command.Execute*",
    "OracleDataAdapter.FillSchema",
    "OracleDataAdapter.Fill",
    "CComboBox.Get*",
    "SqlCeCommand.ExecuteNonQuery",
    "OracleCommand.ExecuteOracleNonQuery",
    "mysqlpp.Query",
    "istream.read*",
    "CListBox.GetText",
    "SqlCeCommand.ExecuteScalar",
    "ifstream.putback",
    "readlink",
    "CHtmlEditCtrl.GetDHtmlDocument",
    "PostThreadMessage",
    "CListCtrl.GetItemText",
    "OracleDataAdapter.Update",
    "OleDbCommand.ExecuteScalar",
    "stdin",
    "SqlDataSource.Delete",
    "OleDbDataAdapter.Fill",
    "fstream.putback",
    "IDbDataAdapter.Fill",
    "_wspawnl",
    "fwprintf",
    "sem_wait",
    "_unlink",
    "ldap_search_ext_sW",
    "signal",
    "PQclear",
    "PQfinish",
    "PQexec",
    "PQresultStatus",
])
