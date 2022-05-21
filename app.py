import math
from flask import Flask,request,jsonify
import numpy as np
import pandas as pd
import pickle
import random
model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

csv1 = pd.read_csv("apps.csv")
csv2 = pd.read_csv("user_reviews.csv")
df = pd.concat([csv1,csv2], ignore_index=True)

@app.route('/')
def index():
    return "Hello world"
@app.route('/predict',methods=['POST'])
def predict():
    App = request.form.get('App')
    try:
        found = df[df['App'].str.contains(App)].iloc[1,16]
        if(math.isnan(found)):
            inp = random.randint(-5,5)
        else:
            inp = found
    except:
           inp = random.randint(-5,5)
    
    input_query = np.array([float(inp)])
    iquery = np.array([input_query])
    
    result = model.predict(iquery)[0]
    if(result==0):
        return jsonify({'Sentiment':'Negative'})
    else:
        return jsonify({'Sentiment':'Positive'})
if __name__ == '__main__':
    app.debug = True
    app.run()