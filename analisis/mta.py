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

lossy_trace = []
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
    i = index
    count = 0
    while trc[i] == 0:
        count += 1
        i += 1
    return count, i

# TO-DO: hacer esto mas eficiente
def find_lossy_trace(trc,index):
    i = index
    # Mientras leemos 1's o la cantidad de ceros es menor a la constante de cambio de estado,
    # nos encontramos en un estado de perdida
    while trc[i] == 1 or zero_counter(trc,i) < c:
        i += 1
    print(f'Lossy state found: {trc[index:i]}')
    return trc[index:i], i
    
ind = 0
while ind < trace.size:
    z_count, new_ind = zero_counter(trace,ind)
    if trace[ind] == 0 and z_count > c:
        error_free_trace.append(trace[ind:new_ind])
        print(f'Error-free state found: {trace[ind:new_ind]}')
        ind = new_ind
        continue

    if trace[ind] == 1: 
        lossy_state, new_ind = find_lossy_trace(trace,ind)
        lossy_trace.append(lossy_state)
        ind = new_ind

print(f'Lossy trace: {lossy_trace}')