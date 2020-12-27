import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Oxigen':2, 'Temparature':9, 'Humidity':6})

print(r.json())