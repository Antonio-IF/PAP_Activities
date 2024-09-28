from fastapi import FastAPI
import uvicorn
import pickle
import numpy as np
import pandas as pd

app = FastAPI()

model = pickle.load(open('models/logistic.pkl', 'rb'))

@app.get('/')
# Example to use http://127.0.0.1:1234/?name=Tony
def greet(name: str):
    return{
        'message': f'Greetings,  {name}!'
    }

@app.get('/health')
def health_check():
    return {
        'status': 'ok'
    }

@app.post('/predict')
def predict(data: list[float]):
    X = [{
        f"X{i+1}": x
        for i, x in enumerate(data)
    }]

    df = pd.DataFrame.from_records(X)
    prediction = model.predict(df)
    return {
        'prediction': int(prediction[0])
    }

if __name__ == '__main__':
    uvicorn.run('app:app', host='127.0.0.1', port=1234, reload=True)
