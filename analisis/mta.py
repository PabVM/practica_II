import numpy as np
import json

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
def count_loss(trc,index):
    i = index
    count = 0
    while trc[i] == 1:
        count += 1
        i += 1
    loss_lengths.append(count)
    return i

ind = 0
while ind < trace.size:
    if trace[ind] == 1:
        ind = count_loss(trace,ind)
    else: ind +=1

mean = np.mean(loss_lengths)
std = np.std(loss_lengths)

print(f'Loss lengths mean: {mean}')
print(f'Standard deviation: {std}')

c = mean + std

print(f'Change-of-state constant C: {c}')



