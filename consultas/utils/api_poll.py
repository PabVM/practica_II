import requests

login = '5ef13a40e833d97984411c43'
password = '4181ccbc3f573d2f77eba1e3e6aad35c'
device_id = "4d6a8b"
response = requests.get(
    f"https://api.sigfox.com/v2/devices/{device_id}/messages",
    auth=(login, password),
    params={"limit": 100}
)

print(response.json()["data"][50])

