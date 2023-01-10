import numpy as np
import json

# Funcion util para hacer las comparaciones
def zfill(string: str, length: int) -> str:
    """Adds zeroes at the begginning of a string 
    until it completes the desired length."""
    return '0' * (length - len(string)) + string

"""
Primero tenemos que crear la traza de los mensajes
"""
with open("experiments_test.json", "r") as file:
    ex = json.load(file)


# Pasamos los mensajes del payload a un arreglo para iterar y crear la traza
payloads = np.sort(np.fromiter(ex, int))
print(payloads)
expected = np.arange(payloads[0],payloads[payloads.size-1])
# La traza debe ser del tamanno del total de mensajes enviados
trace = np.zeros(expected.size)

for i in range(0,expected.size):
    if expected[i] not in payloads:
        trace[i] = 1
print(trace)
"""
Vamos a dejar un arreglo de contadores 
por cada vez que nos encontremos con un 1 en la traza de perdida
"""

loss_lengths = []