import sys
import numpy as np
from scipy.spatial import distance
import time
from collections import defaultdict
import threading
import multiprocessing

class MinhaThread (threading.Thread):

    def __init__(self, train_values, train_labels, test_values, K , name):
        threading.Thread.__init__(self)
        self.train_values = train_values
        self.train_labels = train_labels
        self.K = K
        self.test_values = test_values
        self.counter = 0
        self.test_labels = []
        self.name = name

    def run(self):
    	print('iniciou')
        start_time = time.time()
        self.test_labels = find_labels(self.train_values ,self.train_labels, self.test_values, self.K, self.name)
        print("tempo thread --- %s seconds ---" % (time.time() - start_time))
        print('concluido')

def find_label(train_values, train_labels, sample, K):
	copy_matrix = np.full(train_values.shape, sample, np.float32) #criando matriz com valores do novo ponto
	distances_matrix = (train_values - copy_matrix) ** 2 #calcula a distancia dos pontos de treino com o de teste
	distance_values = distances_matrix.sum(axis = 1) #computa a soma dos valore de cada uma das linhas, retorna um vetor
	count = 0
	min_indexes = [] #lita com o indice dos pontos com menor distancia
	while count < K: #loop que descobre os indices que possuem as menores distancias
		index = np.argmin(distance_values) #recupera o indice do menor valor
		min_indexes.append(index) #adiciona o indice na lista que possui o menor vetor 
		distance_values.itemset(index, False) #adiciona valor sentinela para que nao seja contado novamente
		count += 1

	final_labels = [] #lista com a label referente aos pontos com menor distancia
	for index in min_indexes: 
		final_labels.append(train_labels[index])

	d = defaultdict(int) #codigo que descobre o valor com maior ocorrencia em uma lista , o result eh uma tupla, (valor, numero de ocorrencias)
	for i in final_labels:
	    d[i] += 1
	result = max(d.iteritems(), key=lambda x: x[1])
	return result[0]

def find_labels(train_values, train_labels, samples, K, name):
	labels = []
	for index in xrange(0, samples.shape[0]):
		#print(name + ' index : ' + str(index))
		sample = samples[index, :]
		copy_matrix = np.full(train_values.shape, sample, np.float32) #criando matriz com valores do novo ponto
		distances_matrix = (train_values - copy_matrix) ** 2 #calcula a distancia dos pontos de treino com o de teste
		distance_values = distances_matrix.sum(axis = 1) #computa a soma dos valore de cada uma das linhas, retorna um vetor
		count = 0
		min_indexes = [] #lita com o indice dos pontos com menor distancia
		while count < K: #loop que descobre os indices que possuem as menores distancias
			index = np.argmin(distance_values) #recupera o indice do menor valor
			min_indexes.append(index) #adiciona o indice na lista que possui o menor vetor 
			distance_values.itemset(index, False) #adiciona valor sentinela para que nao seja contado novamente
			count += 1

		final_labels = [] #lista com a label referente aos pontos com menor distancia
		for index in min_indexes: 
			final_labels.append(train_labels[index])

		d = defaultdict(int) #codigo que descobre o valor com maior ocorrencia em uma lista , o result eh uma tupla, (valor, numero de ocorrencias)
		for i in final_labels:
		    d[i] += 1
		result = max(d.iteritems(), key=lambda x: x[1])
		labels.append(result[0])
	return labels

def get_train_values(file_path, divisor = ' '):
	print('iniciando leitura arquivo')
	start_time = time.time()
	X = [] #lista com os valores
	y = [] #lista com as labels
	file = open(file_path, 'r')
	file.readline()
	for line in file:
		values = line.split(divisor) #divide o arquivo pelo caractere X
		[float(i) for i in values] #converte valores para float
		y.append(int(values.pop(len(values) - 1))) #isola o valor referente a label
		X.append(values)
	file.close()
	X = np.array( X, np.float32)
	print("arquivo treino --- %s seconds ---" % (time.time() - start_time))
	return X, y

def get_test_values(file_path, divisor = ' '):
	test_values_arrays= []
	X = []
	y = []
	file = open(file_path, 'r')
	file.readline()
	for line in file:
		values = line.split(divisor) #divide o arquivo pelo caractere X
		[float(i) for i in values] #converte valores para float
		y.append(int(values.pop(len(values) - 1))) #isola o valor referente a label
		X.append(values)
	file.close()
	cpus = multiprocessing.cpu_count() #descobre numero de processadores
	ini = 0
	slice_size = len(X) / cpus
	end = slice_size
	x = 0
	while x < cpus - 1:
		test_values_arrays.append(X[ini:end])
		x += 1
		ini += slice_size
		end += slice_size
	test_values_arrays.append(X[ini:])
	X = []
	for values in test_values_arrays:
		X.append(np.array(values, np.float32))
		print(len(values))
	return X, y



train_path = sys.argv[1]
test_path = sys.argv[2]
K = int(sys.argv[3])


train_values, train_labels = get_train_values(train_path)
#test_values, test_labels = get_train_values(test_path)
test_values, test_labels = get_test_values(test_path)

main_time = time.time()
threads = []
x = 0
for values in test_values:
	thread = MinhaThread(train_values, train_labels, values, K, str(x))	
	thread.start()
	threads.append(thread)
	x += 1


for t in threads:
	#t.start()
	t.join()


'''
main_time = time.time()
labels = []
counter = 0 
while counter < test_values.shape[0]:
	label = find_label(train_values, train_labels, test_values[counter, :], K)
	labels.append(label)
	#print(counter)
	counter += 1
'''


print('SEGUINDO FLUXO')
print(threads[0].test_labels + threads[1].test_labels + threads[2].test_labels + threads[3].test_labels)
#print(labels)
#print(test_labels[:60])
print("arquivo treino --- %s seconds ---" % (time.time() - main_time))
#print(threads[1].test_labels)
#print(threads[2].test_labels)
#print(threads[3].test_labels)
