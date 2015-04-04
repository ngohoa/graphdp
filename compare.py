import networkx as nx 
import time
import random
import math
import numpy as np

def read_data(f_name):
	graph = nx.Graph()
	with open(f_name) as f:
		for line in f:
			head = line.split(" ")[0]
			tail = line.split(" ")[1]
			# print head, tail
			graph.add_edge(head, tail)
	return graph

def compare(f1, f2):
	g1 = read_data(f1)
	g2 = read_data(f2)
	avg1, avg2 = 0, 0
	for node in g1.nodes():
		avg1 += g1.degree(node)
	for node in g2.nodes():
		avg2 += g2.degree(node)
	print "Average degree......:       ", avg1*1.0/(len(g1.nodes())), avg2*1.0/(len(g2.nodes()))

	trg1, trg2 = 0, 0
	tr_list1 = list(nx.triangles(g1).values())
	tr_list2 = list(nx.triangles(g2).values())
	print "Triangles......:       ", sum(tr_list1)/3, sum(tr_list2)/3


compare('ordered_edges_dblp.dat', 'results_edges_small.dat')
# compare('ordered_edges_small.dat', 'results_edges_small.dat')
# compare('ordered_edges_small.dat', 'Lungtung/txt_fb_3_1_15.txt')