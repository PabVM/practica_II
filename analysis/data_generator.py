import json
import random

data = {}

def zfill(string: str, length: int) -> str:
    """Adds zeroes at the begginning of a string 
    until it completes the desired length."""
    return '0' * (length - len(string)) + string

for i in range(30000):
    data[i] = i

for i in range(13634):
    data.pop(random.choice(list(data.keys())))

data = {k: v for k, v in sorted(data.items(), key=lambda x: x[1])}
with open('generated_data.json','w') as json_file:
    json.dump(data,json_file,indent=2)