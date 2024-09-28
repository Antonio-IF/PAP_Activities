import requests

data = list(range(23))
response = requests.post('http://127.0.0.1:1234/predict', json=data)

prediction = response.json()['prediction']

# Cada vez que lleguen 1000 nuevos datosw hacemos las pruebas de hipotesis, chi cuadrada y 
# ks para reentrenar

# dataset de pred con mi Y y el link de un github, poner gitignore de python. 
