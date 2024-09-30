import requests

# --------------------------------------------------

# API ENDPOINT
url = 'http://127.0.0.1:5000/predict'

# DATA TO SEND IN THE REQUEST (NEW DATA FOR PREDICTION)
data = {
    "X1": 50000, "X2": 1, "X3": 2, "X4": 3, "X5": 45, 
    "X6": 0, "X7": 0, "X8": 0, "X9": 0, "X10": 0, 
    "X11": 0, "X12": 0, "X13": 0, "X14": 0, "X15": 0, 
    "X16": 0, "X17": 0, "X18": 0, "X19": 0, "X20": 0, 
    "X21": 0, "X22": 0, "X23": 0
}

# MAKE A POST REQUEST TO THE API
response = requests.post(url, json=data)

# ------------------------------

# CHECK IF THE REQUEST WAS SUCCESSFUL
if response.status_code == 200:
    print("Prediction:", response.json())

# HANDLE ERROR CASES
else:
    print(f"Error: {response.status_code}, {response.text}")