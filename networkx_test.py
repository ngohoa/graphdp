import networkx as nx 
import time
import random
import math

k = 3
e = 20
maxmetric = 2
minmetric = 0

def read_data(f_name):
	graph = nx.Graph()
	with open(f_name) as f:
		for line in f:
			head = line.split(" ")[0]
			tail = line.split(" ")[1]
			# print head, tail
			graph.add_edge(head, tail)
	return graph

def mutual_friend(graph, node1, node2):
	n1 = graph.neighbors(node1)
	n2 = graph.neighbors(node2)
	return len(set(n1).intersection(set(n2)))

def mutual_friends_all(graph):
	nodes = graph.nodes()
	maxmf, minf = 0, 10000000
	for node1 in nodes:
		for node2 in nodes:
			if (node2 != node1):
				num = mutual_friend(graph, node1, node2)
				if num > maxmf:
					maxmf = num
				if num < minf:
					minf = num
	return minf, maxmf

def max_degree(graph):
	maxd = 0
	for node in graph.nodes():
		if graph.degree(node) > maxd:
			maxd = graph.degree(node)
	return maxd

def add(nodes_list, ver):
	flag = 0
	for node in nodes_list:
		if node[0] == ver[0]:
			flag = 1
			if ver[1] < node[1]:
				node[1] = ver[1]
	if flag == 0:
		nodes_list.append(ver)

def not_in(nodes_list, ver):
	for node in nodes_list:
		if node[0] == ver[0]:
			return False
	return True

# for one node
def neighbours_k_nodes(graph, start, k):
	visited, nodes_list = [], [[start, 0]]
	while nodes_list:
		node_read = nodes_list.pop()
		if not_in(visited, node_read):
			node_neighbours = graph.neighbors(node_read[0])
			for neighbour in node_neighbours:
				if not_in(nodes_list, [neighbour, node_read[1]+1]) and node_read[1]+1 < k:
					nodes_list.append([neighbour, node_read[1]+1])
		add(visited, node_read)
	return visited

# for one node
def random_select_nodes(nodes_list):
	real_edges, virtual_edges = [], []
	for node in nodes_list:
		if node[1] == 1:
			real_edges.append(node)
		if node[1] > 1:
			virtual_edges.append(node)
	if len(real_edges) < len(virtual_edges):
		return real_edges, random.sample(virtual_edges, len(real_edges))
	else:
		return real_edges, virtual_edges

def scale(matrix, min_old, max_old, min_new, max_new):
	for edge in matrix:
		edge[3] = (max_new - min_new)/(max_old - min_old) * (edge[3] - min_old) + min_new

# for one node
def metric_computing(graph, ver, nodes_list, maxmetric, minmetric, o):
	matrix = []
	for node in nodes_list:
		m = mutual_friend(graph, ver, node[0])
		m = (1 - math.exp(-1.0/o))/(maxmetric - minmetric) * (m - minmetric) + math.exp(-1/o)
		matrix.append([ver, node[0], node[1], m])
	return matrix

# for one node
def penalty_computing(graph, ver, nodes_list, option):
	matrix = []
	if option == 1:
		for node in nodes_list:
			matrix.append([ver, node[0], node[1], 1])
	if option == 2:
		for node in nodes_list:
			matrix.append([ver, node[0], node[1], mutual_friend(graph, ver, node[0])])
	return matrix

# for whole graph
def a_setting(graph, matrix_real, matrix_virtual, penalty_real, penalty_virtual):
	a1, a2, c1, c2 = 0, 0, 0, 0
	i = 0
	for edge in matrix_real:
		# if edge[2] == 1:
		a1 += edge[3]*penalty_real[i][3]
		c1 += penalty_real[i][3]
		i += 1
	i = 0
	for edge in matrix_virtual:
		# if edge[2] > 1:
		a2 += edge[3]*penalty_virtual[i][3]
		c2 += penalty_virtual[i][3]
		i += 1
	return a1, a2, c1, c2

# for whole graph
def parameter_setting(a1, a2, c1, c2, matrix_real, matrix_virtual, penalty_real, penalty_virtual, o):
	if a1 < a2: # scale a1
		if a1 < math.exp(-1.0/o)*c2:
			return 0
		else:
			no_edge = 0
			i = 0
			for edge in matrix_virtual:
				# if edge[2] > 1:
				no_edge = no_edge + (edge[3] - math.exp(-1.0/o))*penalty_virtual[i][3]
				i += 1
			a = (1-math.exp(-1.0/o))*((a1-math.exp(-1.0/o)*c2) / no_edge) + math.exp(-1.0/o)
			print a
			scale(matrix_virtual, math.exp(-1.0/o), 1, math.exp(-1.0/o), a)
	else:
		if a2 < math.exp(-1.0/o)*c1:
			return 0
		else:
			have_edge = 0
			i = 0
			for edge in matrix_real:
				# if edge[2] == 1:
				have_edge = have_edge + (edge[3] - math.exp(-1.0/o))*penalty_real[i][3]
				i += 1
			a = (1-math.exp(-1.0/o))*((a2-math.exp(-1.0/o)*c1) / have_edge) + math.exp(-1.0/o)
			print a
			scale(matrix_real, math.exp(-1.0/o), 1, math.exp(-1.0/o), a)
	matrix = matrix_real + matrix_virtual
	for edge in matrix:
		edge[3] = (-o)*(math.log(edge[3], math.e))
	return matrix

# def noise_adding(matrix, threshold):
# 	for edge in matrix:

if __name__ == "__main__":

	graph = read_data('test_dataset.dat')
	# print mutual_friend(graph, '1', '2')
	t0 = time.clock()
	matrix_real, matrix_virtual, penalty_real, penalty_virtual = [], [], [], []
	# print graph.nodes()
	o = -1/(math.log(2/(math.exp(2*e/k/(k-1)) + 1), math.e))
	i = 0
	for node in graph.nodes():
		# i += 1
		bsf_nodes = neighbours_k_nodes(graph, node, k)
		# print bsf_nodes
		real, virtual = random_select_nodes(bsf_nodes)
		print node, '.....', real, virtual

		m_real = metric_computing(graph, node, real, maxmetric, minmetric, o)
		m_virtual = metric_computing(graph, node, virtual, maxmetric, minmetric, o)
		p_real = penalty_computing(graph, node, real, 1)
		p_virtual = penalty_computing(graph, node, virtual, 1)

		matrix_real += m_real
		matrix_virtual += m_virtual
		penalty_real += p_real
		penalty_virtual += p_virtual
		# print matrix_real, matrix_virtual

	a1, a2, c1, c2 = a_setting(graph, matrix_real, matrix_virtual, penalty_real, penalty_virtual)
	print a1, a2, c1, c2
	matrix = parameter_setting(a1, a2, c1, c2, matrix_real, matrix_virtual, penalty_real, penalty_virtual, o)
	# print matrix

	print time.clock() - t0, "seconds process time"