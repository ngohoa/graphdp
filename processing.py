import networkx as nx 
import time
import random
import math
import numpy as np

k = 3
e = 50
MAX = 213
MIN = 0

def read_data(f_name):
	graph = nx.Graph()
	with open(f_name) as f:
		for line in f:
			head = line.split(" ")[0]
			tail = line.split(" ")[1]
			# print head, tail
			graph.add_edge(head, tail)
	return graph

def read_matrix(fn):
	save = []
	with open(fn, 'r') as f:
		for line in f:
			n = line.split(" ")[0]
			v = line.split(" ")[1]
			save.append([n, v])
	return save

def mutual_friend(graph, node1, node2):
	n1 = graph.neighbors(node1)
	n2 = graph.neighbors(node2)
	return len(set(n1).intersection(set(n2)))

def scale(matrix, min_old, max_old, min_new, max_new):
	for i in range (0, len(matrix)):
		matrix[i] = (max_new - min_new)/(max_old - min_old) * (matrix[i] - min_old) + min_new

def penalty_computing(graph, save, option):
	penalty = []
	if option == 1:
		for edge in save:
			penalty.append(1)
	if option == 2:
		for edge in save:
			penalty.append(mutual_friend(graph, edge[0], edge[1]))
	return penalty

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

if __name__ == "__main__":
	# graph = read_data('test_dataset.dat')
	graph = read_data('ordered_edges_dblp.dat')
	# graph = read_data('ordered_edges_big.dat')
	t0 = time.clock()
	o = -1.0/(math.log(2.0/(math.exp(2.0*e/k/(k-1)) + 1), math.e))
	save_real = read_matrix('Saves/save_real.dat')
	save_virtual = read_matrix('Saves/save_virtual.dat')
	matrix_real = []
	matrix_virtual = []

	for save in save_real:
		m = mutual_friend(graph, save[1], save[0])
		m = (1 - math.exp(-1.0/o))/(MAX - MIN) * (m - MIN) + math.exp(-1.0/o)
		matrix_real.append(m)

	for save in save_virtual:
		m = mutual_friend(graph, save[1], save[0])
		m = (1 - math.exp(-1.0/o))/(MAX - MIN) * (m - MIN) + math.exp(-1.0/o)
		matrix_virtual.append(m)

	penalty_real = penalty_computing(graph, save_real, 2)
	penalty_virtual = penalty_computing(graph, save_virtual, 2)

	a1, a2, c1, c2 = a_setting(graph, matrix_real, matrix_virtual, penalty_real, penalty_virtual)
	mr, mv = parameter_setting(a1, a2, c1, c2, matrix_real, matrix_virtual, penalty_real, penalty_virtual, o)
	
	fn = 'results_edges_dblp.dat'
	print "Removing edges......:     ", noise_adding_real(mr, save_real, 0, o, 0, fn)
	print "Adding edges........:     ", noise_adding_virtual(mv, save_virtual, 0, o, 0, fn)
	print time.clock() - t0, "seconds process time"