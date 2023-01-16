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
                print(f'Lossy state begins at: {start}')
                print(f'Lossy state ends at: {end}')
                lossy_trace = np.concatenate((lossy_trace, np.array(lossy_state)))
                print(f'Lossy state found: {lossy_state}')
                end += count
                start = end

print(f'Lossy trace: {lossy_trace}')
print(f'Error free trace: {error_free_trace}')
