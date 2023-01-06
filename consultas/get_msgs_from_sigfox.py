# Este codigo es para realizar consultas cada cierta cantidad de tiempo al backend de Sigfox
# y verificar que los numeros de secuencia no se repitan

import requests
import time
import json

from utils.casting import hex_to_bytes

seq_nums = []
data = {}

data_json = {}

with open("credentials.json") as credentials_file:
    credentials = json.load(credentials_file)

login = credentials["login"]
password = credentials["password"]
device_id = credentials["device_id"]

# Hay que ver alguna forma de guardar el diccionario en un csv o algun formato que permita despues procesar los datos

def add_msg(seq, payload, dict):
    if seq in dict and dict[seq] != payload:
        # Si el numero de secuencia ya existe y tiene asignado otro payload
        add_msg(seq + 1, payload, dict)
    else:
        dict[seq] = payload


while True:
    response = requests.get(
        f"https://api.sigfox.com/v2/devices/{device_id}/messages",
        auth=(login, password),
        params={"limit": 100}
    )
    # Se agregan los payloads asociados a su número de secuencia en el diccionario data
    for elem in response.json()["data"]:
        payload = hex_to_bytes(elem["data"]).decode()
        add_msg(elem["seqNumber"], payload, data)
    print(data)
    data_json.update(data)
    with open('experiments_test.json', 'w') as json_file:
        json.dump(data_json,json_file,indent=2)
    time.sleep(2000)

