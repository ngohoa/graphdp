import networkx as nx 
import time
import random
import math
import numpy as np

K = 3
e = 30
B = 1

## ordered_edges_small 
# MAX = 293
# MIN = 0

## ordered_edges_big
# MAX = 265
# MIN = 0

## ordered_edges_dblp
# MAX = 213
# MIN = 0

## youtube
MAX = 4034
MIN = 0

def read_data(f_name):
	graph = nx.Graph()
	with open(f_name) as f:
		for line in f:
			# print line.split(" ")
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
	# visited, nodes_list = [], [[start, 0]]
	# while nodes_list:
	# 	node_read = nodes_list.pop()
	# 	if not_in(visited, node_read):
	# 		node_neighbours = graph.neighbors(node_read[0])
	# 		for neighbour in node_neighbours:
	# 			if not_in(nodes_list, [neighbour, node_read[1]+1]) and node_read[1]+1 < k:
	# 				nodes_list.append([neighbour, node_read[1]+1])
	# 	add(visited, node_read)
	# return visited

	adjacent = []
	adjacent_loop = []
	non_adjacent = []
	# count = 0
	for node in graph.neighbors(start):
		adjacent_loop.append(node)
		if int(start) < int(node):
			adjacent.append(node)
		# add(visited, [node, 1])
	for node in adjacent_loop:
		non_adjacent = list(set(non_adjacent) | set(graph.neighbors(node)))
	non_adjacent = list(set(non_adjacent) - set(adjacent_loop) - set([start]))
		# for n in graph.neighbors(node):
			# add(visited, [n, 2])
	if B*len(adjacent_loop) < len(non_adjacent):
	 	return adjacent, random.sample(non_adjacent, B*len(adjacent_loop))
	return adjacent, non_adjacent

# for one node
def random_select_nodes(real_edges, virtual_edges):
	# real_edges, virtual_edges = [], []
	# for node in nodes_list:
	# 	if node[1] == 1:
	# 		real_edges.append(node)
	# 	if node[1] > 1:
	# 		virtual_edges.append(node)

	if len(real_edges) < len(virtual_edges):
		return real_edges, random.sample(virtual_edges, len(real_edges))
	else:
		return real_edges, virtual_edges

def scale(matrix, min_old, max_old, min_new, max_new):
	for i in range (0, len(matrix)):
		matrix[i] = (max_new - min_new)/(max_old - min_old) * (matrix[i] - min_old) + min_new

# for one node
def metric_computing(graph, ver, real_edges, virtual_edges, maxmetric, minmetric, o):
	real_matrix = []
	virtual_matrix = []
	for node in real_edges:
		m = mutual_friend(graph, ver, node)
		m = (1 - math.exp(-1.0/o))/(maxmetric - minmetric) * (m - minmetric) + math.exp(-1.0/o)
		real_matrix.append(m)
	for node in virtual_edges:
		m = mutual_friend(graph, ver, node)
		m = (1 - math.exp(-1.0/o))/(maxmetric - minmetric) * (m - minmetric) + math.exp(-1.0/o)
		virtual_matrix.append(m)
	return real_matrix, virtual_matrix

# for one node
def penalty_computing(graph, ver, real_edges, virtual_edges, option):
	real_matrix = []
	virtual_matrix = []
	if option == 1:
		for node in real_edges:
			real_matrix.append(1)
		for node in virtual_edges:
			virtual_matrix.append(1)
	if option == 2:
		for node in real_edges:
			real_matrix.append(mutual_friend(graph, ver, node))
		for node in virtual_edges:
			virtual_matrix.append(mutual_friend(graph, ver, node))
	return real_matrix, virtual_matrix

# for whole graph
def a_setting(graph, matrix_real, matrix_virtual, penalty_real, penalty_virtual):
	a1, a2, c1, c2 = 0, 0, 0, 0
	i = 0
	for edge in matrix_real:
		# if edge[2] == 1:
		a1 += edge*penalty_real[i]
		c1 += penalty_real[i]
		i += 1
	i = 0
	for edge in matrix_virtual:
		# if edge[2] > 1:
		a2 += edge*penalty_virtual[i]
		c2 += penalty_virtual[i]
		i += 1
	return a1, a2, c1, c2

# for whole graph
def parameter_setting(a1, a2, c1, c2, matrix_real, matrix_virtual, penalty_real, penalty_virtual, o):
	# print "ok1"
	if a1 < a2: # scale a1
		# print "ok2"
		if a1 < math.exp(-1.0/o)*c2:
			return 0
		else:
			no_edge = 0
			i = 0
			for edge in matrix_virtual:
				# if edge[2] > 1:
				no_edge = no_edge + (edge - math.exp(-1.0/o))*penalty_virtual[i]
				i += 1
			a = (1-math.exp(-1.0/o))*((a1-math.exp(-1.0/o)*c2) / no_edge) + math.exp(-1.0/o)
			# print a
			scale(matrix_virtual, math.exp(-1.0/o), 1, math.exp(-1.0/o), a)
	else:
		# print "ok3"
		if a2 < math.exp(-1.0/o)*c1:
			# print "ok4"
			return 0
		else:
			# print "ok5"
			have_edge = 0
			i = 0
			for edge in matrix_real:
				# if edge[2] == 1:
				have_edge = have_edge + (edge - math.exp(-1.0/o))*penalty_real[i]
				i += 1
			a = (1-math.exp(-1.0/o))*((a2-math.exp(-1.0/o)*c1) / have_edge) + math.exp(-1.0/o)
			# print a
			scale(matrix_real, math.exp(-1.0/o), 1, math.exp(-1.0/o), a)
	# matrix = matrix_real + matrix_virtual
	
	for i in range (0, len(matrix_real)):
		matrix_real[i] = (-o)*(math.log(matrix_real[i], math.e))
	for i in range (0, len(matrix_virtual)):
		matrix_virtual[i] = (-o)*(math.log(matrix_virtual[i], math.e))
	return matrix_real, matrix_virtual

def noise_adding_real(matrix, save, threshold, o , u, fn):
	count = 0
	i = 0
	with open(fn, 'w') as f:
	 	for edge in matrix:
	 		noise = np.random.laplace(u, o)
	 		if edge + noise < threshold:
	 			count += 1
	 		else:
	 			f.write(save[i][0] + " " + save[i][1] + " \n")
	 		i += 1
 	return count

def noise_adding_virtual(matrix, save, threshold, o , u, fn):
	count = 0
	i = 0
	with open(fn, 'a') as f:
	 	for edge in matrix:
	 		noise = np.random.laplace(u, o)
	 		if edge + noise < threshold:
	 			count += 1
	 			f.write(save[i][0] + " " + save[i][1] + " \n")
	 		i += 1
 	return count

def min_e(matrix_real, matrix_virtual, penalty_real, penalty_virtual):
	maxmetric = 1
	c1 = sum(penalty_virtual)
	c2 = 0
	y1 = 0
	nk = K*(K-1)/2
	for i in range(0, matrix_real):
		c2 += (1-matrix_real[i]*1.0/maxmetric)*penalty_real[i]
		y1 += (matrix_real[i]*1.0/maxmetric)*penalty_real[i]
	x1 = c1 - c2
	print "x1.......:     ", x1
	e_min = nk*math.log(2*x1/y1 - 1, math.e)
	print "e_min.......:", e_min

	c3 = sum(penalty_real)
	c4 = 0
	y2 = 0
	for i in range(o, matrix_virtual):
		c4 += (1-matrix_virtual[i]*1.0/maxmetric)*penalty_virtual[i]
		y2 += (matrix_virtual[i]*1.0/maxmetric)*penalty_virtual[i]
	x2 = c3 - c4
	print "x2.......:     ", x2
	e_min = nk*math.log(2*x2/y2 - 1, math.e)
	print "e_min.......:", e_min

if __name__ == "__main__":
	# graph = read_data('test_dataset.dat')
	# graph = read_data('ordered_edges_small.dat')
	graph = read_data('youtube.txt')
	# graph = read_data('ordered_edges_dblp.dat')
	# graph = read_data('ordered_edges_big.dat')
	# print mutual_friend(graph, '1', '2')
	t0 = time.clock()
	matrix_real, matrix_virtual, penalty_real, penalty_virtual = [], [], [], []
	# print graph.nodes()
	o = -1.0/(math.log(2.0/(math.exp(2.0*e/K/(K-1)) + 1), math.e))
	check = []
	# node = '1'
	# if node == '1':
	save_real = []
	save_virtual = []
	for node in graph.nodes():
		# temp = []
		real, virtual = neighbours_k_nodes(graph, node, K)
		# virtual = list(set(virtual) -set(check))
		# print bsf_nodes
		# real, virtual = random_select_nodes(real_edges, virtual_edges)

		# for ver in virtual:
		# 	if int(ver) < int(node):
		# 		if [ver, node] not in check:
		# 			check.append([ver, node])
		# 		else:
		# 			temp.append(ver)
		# 	else:
		# 		if [node, ver] not in check:
		# 			check.append([node, ver])
		# 		else:
		# 			temp.append(ver)
		# virtual = list(set(virtual) - set(temp))

		# for n in real:
		# 	mut = mutual_friend(graph, node, n)
		# 	if mut < MIN:
		# 		MIN = mut
		# 	if mut > MAX:
		# 		MAX = mut

		# for n in virtual:
		# 	mut = mutual_friend(graph, node, n)
		# 	if mut < MIN:
		# 		MIN = mut
		# 	if mut > MAX:
		# 		MAX = mut
	# print MIN, MAX

		# if len(virtual) > B*len(real):
		# 	virtual = random.sample(virtual, B*len(real))
		for ver in real:
			save_real.append([node, ver])
		for ver in virtual:
			save_virtual.append([node, ver])
		# # print node, '.....', real, virtual

		# m_real, m_virtual = metric_computing(graph, node, real, virtual, MAX, MIN, o)
		# # print len(m_real), len(m_virtual)
		# # print max(m_real), min(m_real), max(m_virtual), min(m_virtual)
		# p_real, p_virtual = penalty_computing(graph, node, real, virtual, 1)f

		# matrix_real += m_real
		# matrix_virtual += m_virtual
		# penalty_real += p_real
		# penalty_virtual += p_virtual
	


	# min_e(matrix_real, matrix_virtual, penalty_real, penalty_virtual)

	print time.clock() - t0, "seconds process time"

	fr = open('Saves/save_real_youtube.dat', 'w')
	fv = open('Saves/save_virtual_youtube.dat', 'w')
	for i in range (0, len(save_real)):
		fr.write(save_real[i][0] + " " + save_real[i][1] + " \n")
	for i in range (0, len(save_virtual)):
		fv.write(save_virtual[i][0] + " " + save_virtual[i][1] + " \n")
	fr.close()
	fv.close()

	# a1, a2, c1, c2 = a_setting(graph, matrix_real, matrix_virtual, penalty_real, penalty_virtual)
	# mr, mv = parameter_setting(a1, a2, c1, c2, matrix_real, matrix_virtual, penalty_real, penalty_virtual, o)
	
	# fn = 'results_edges_youtube.dat'
	# print "Removing edges......:     ", noise_adding_real(mr, save_real, 0, o, 0, fn)
	# print "Adding edges........:     ", noise_adding_virtual(mv, save_virtual, 0, o, 0, fn)
	print time.clock() - t0, "seconds process time"