#!/usr/bin/env python3
import numpy as np
num_graphs = 0
graphs = []
labels = []
def pre_process():
    with open("yeast", 'r') as file:
        lines = file.readlines()
        
    current_line=0
    while current_line<len(lines):
        single_graph={}
        while lines[current_line].startswith("#"):
             current_line += 1  
        num_nodes = int(lines[current_line])
        current_line += 1
        nodes = []
        for i in range(num_nodes):
            label=lines[current_line].strip()
            nodes.append(label)
            current_line += 1
        single_graph["vertx"]=nodes
        num_edges = int(lines[current_line])
        current_line += 1
        edges = []
        for i in range(num_edges):
            edge_info = lines[current_line].strip().split()
            source = int(edge_info[0])
            dest = int(edge_info[1])
            label = int(edge_info[2])
            edges.append([source, dest, label])
            current_line += 1
        single_graph["edges"]=edges
        graphs.append(single_graph)
        while current_line<len(lines) and lines[current_line]=='\n':
             current_line += 1
    with open("yeast_processed", "w") as outfile:
        outfile.write(f"# {len(graphs)}\n")
        label_map={}
        map_int=0
        for i in range(len(graphs)):
            graph = graphs[i]
            outfile.write(f"t # {i}\n")
            node_count=0
            for label in graph["vertx"]:
                mapd_val=label_map.get(label,"NONE")
                if mapd_val=='NONE':
                   label_map[label]=map_int
                   map_int+=1
                x=label_map.get(label)
                outfile.write(f"v {node_count} {x}\n")
                node_count+=1
            for edge in graph["edges"]:
                outfile.write(f"e {edge[0]} {edge[1]} {edge[2]}\n")
pre_process()