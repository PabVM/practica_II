import numpy as np
import json
import math
import matplotlib.pyplot as plt
from itertools import product, permutations

"""
Primero tenemos que crear la traza de los mensajes
"""
with open("experiments_test.json", "r") as file:
    ex = json.load(file)

# Pasamos los mensajes del payload a un arreglo para iterar y crear la traza
iterable = (p for p in ex.values())
payloads = np.sort(np.fromiter(iterable, int))
print(payloads)
expected = np.arange(start=payloads[0],stop=payloads[payloads.size-1]+1,dtype=int)
print(expected)
# La traza debe ser del tamanno del total de mensajes enviados
trace = np.zeros(expected.size, dtype=int)

for i in range(0,expected.size):
    if expected[i] not in payloads:
        trace[i] = 1
print(trace)

"""
Vamos a dejar un arreglo de contadores 
por cada vez que nos encontremos con un 1 en la traza de perdida
"""

loss_lengths = []

"""
Se recorre el arreglo correspondiente a la traza y cada vez que pille un 1,
inicializar un contador, dejar que corra hasta encontrar un cero y luego hacerle append a loss_lenghts
"""
def count_loss(trc,index,lengths):
    i = index
    count = 0
    while trc[i] == 1 and i < trc.size-1:
        count += 1
        i += 1
    lengths.append(count)
    return i

ind = 0
while ind < trace.size:
    if trace[ind] == 1:
        ind = count_loss(trace,ind,loss_lengths)
    else: ind +=1

mean = np.mean(loss_lengths)
std = np.std(loss_lengths)

print(f'Loss lengths mean: {mean}')
print(f'Standard deviation: {std}')

c = mean + std

print(f'Change-of-state constant C: {c}')

lossy_trace = np.array([])
error_free_trace = []

"""
Ahora corresponde crear la traza de perdida

Para esto, crearemos una subrutina que extraiga los estados de error
y los concatene en lossy_trace

En este caso creo que no deberia ser relevante el obtener la traza libre de errores,
pero la vamos a sacar igual

"""
# Necesito un contador de ceros
def zero_counter(trc,index):
    count = 0
    while trc[index] == 0 and index < trc.size-1:
        count += 1
        index += 1
    return count

# ¿En qué momentos hay que contar ceros?
# A: Cuando en la traza pillo un 1 y luego un 0
# Voy a necesitar dos índices para indicar el inicio y el fin del segmento
# Entonces puedo avanzar hasta encontrar un 1, guardar ese índice de inicio, seguir hasta encontrar un cero y contar
# Si la cantidad de 0's es menor a C, se actualiza el índice de fin de segmento y se sigue iterando
# Si la cantidad de 0's es mayor o igual a C, se deja el índice de fin de segmento como estaba y se agrega el segmento a lossy_trace

start = 0
end = 0
while start < trace.size-1 and end < trace.size-1:
    if trace[start] == 0 and start == end:
        error_free_trace.append(trace[start])
        start += 1
        end += 1
        continue
    elif trace[start] == 1:
        # Actualizo end
        end += 1
        # Si me topo con un cero, empiezo a contar
        if trace[end] == 0:
            count = zero_counter(trace,end)
            # Si la cantidad de 0's consecutivos es mayor a C, se corre start
            if count < c:
                end += count
                continue
            else: 
                lossy_state = trace[start:end]
                #print(f'Lossy state begins at: {start}')
                #print(f'Lossy state ends at: {end}')
                lossy_trace = np.concatenate((lossy_trace, np.array(lossy_state)))
                #print(f'Lossy state found: {lossy_state}')
                end += count
                start = end

print(f'Lossy trace: {lossy_trace}')
print(f'Error free trace: {error_free_trace}')

def runs_test(trc):
    window_size = 50
    print(f'Lossy trace length: {trc.size}')
    partitions = math.floor((trc.size)/window_size)
    print(f'Number of partitions: {partitions}')
    trc_partitioned = np.array_split(trc,partitions)
    runs = np.array([])
    for prtn in trc_partitioned:
        i = 0
        sub_runs = []
        while i < prtn.size-1:
            if prtn[i] == 1:
                i = count_loss(prtn,i,sub_runs)
            else: i += 1
        runs = np.concatenate((runs,np.array(sub_runs)))
    median = np.median(runs)
    runs_above = np.where(runs > median)
    print(f'Number of runs above median: {runs_above[0].size}')
    runs_below = np.where(runs < median)
    print(f'Number of runs below median: {runs_below[0].size}')
    # TO-DO: ver bien qué es lo que se tiene que histogramear
    plt.hist(runs,bins=10)
    plt.show()
    
runs_test(lossy_trace)

def count_appearances(subtrc,spltd_trc):
    count = 0
    for element in spltd_trc:
        if np.array_equal(subtrc,element):
            count += 1
    return count

def values_to_list(dict):
    l = np.array([],dtype=int)
    for v in dict.values():
        l = np.append(l,int(v))
    return l

def get_indexes(trc,element):
    indexes = np.array([],dtype=int)
    i = 0
    while i < len(trc):
        if np.array_equal(trc[i],element):
            indexes = np.append(indexes,i)
        i += 1
    return indexes

def count_appearances_followed_by(num,trc_spltd,permutation):
    count = 0
    indxs = get_indexes(trc_spltd,permutation)
    # Vamos recorriendo el arreglo de índices y revisando el primer número del elemento siguiente
    i = 0
    while i < indxs.size and indxs[i] < len(trc_spltd) - 1:
        index = indxs[i]
        if trc_spltd[index+1][0] == num:
            count += 1
        i += 1
    return count

def get_conditional_entropy(trc, order):
    n_partitions = math.floor((trc.size)/order)
    # Primero, particionamos la traza para poder hacer las búsquedas necesarias
    trc_splitted = np.array_split(trc,n_partitions)
    # Calculamos todas las posibles permutaciones del largo del orden
    perm = list(product([0,1], repeat=order))
    print(f'Possible permutations: {perm}')
    # Y procedemos a contar las ocurrencias
    appearances = {}
    
    # Por cada permutación, contamos el número de ocurrencias y lo guardamos en un diccionarios
    for element in perm:
        count = count_appearances(element,trc_splitted)
        appearances[element] = count
    
    # Ahora debemos buscar las ocurrencias de cada elemento de las permutaciones (que aparezcan)
    # para definir los factores de la segunda suma
    # Es decir:
    num_of_ap_fllwd_by_0 = {}
    num_of_ap_fllwd_by_1 = {}

    sum_0_arr = np.array([])
    sum_1_arr = np.array([])

    for k,v in appearances.items():
        if v != 0:
            num_of_ap_fllwd_by_0[k] = count_appearances_followed_by(0,trc_splitted,k)
            num_of_ap_fllwd_by_1[k] = count_appearances_followed_by(1,trc_splitted,k)

            first_factor_fllwd_by_0 = num_of_ap_fllwd_by_0[k]/n_partitions
            second_factor_fllwd_by_0 = np.log2(num_of_ap_fllwd_by_0[k]/v) if num_of_ap_fllwd_by_0[k] != 0 else 0
            sum_0_arr = np.append(sum_0_arr, (first_factor_fllwd_by_0*second_factor_fllwd_by_0))


            first_factor_fllwd_by_1 = num_of_ap_fllwd_by_1[k]/n_partitions
            second_factor_fllwd_by_1 = np.log2(num_of_ap_fllwd_by_1[k]/v) if num_of_ap_fllwd_by_1[k] != 0 else 0
            sum_1_arr = np.append(sum_1_arr,(first_factor_fllwd_by_1*second_factor_fllwd_by_1))
    
    sum_0 = np.sum(sum_0_arr)
    sum_1 = np.sum(sum_1_arr)

    return -(sum_0+sum_1)

    

# Luego de esto, debemos calcular la entropía para distintos órdenes (vamos a tomar un máximo de 6 porque 2**6 estados como máximo suena razonable)
entropies = {}
order = 1
while order < 6:
    print(f'Current order:{order}')
    entropies[order] = get_conditional_entropy(lossy_trace,order)
    order += 1

print(f'Entropies: {entropies}')