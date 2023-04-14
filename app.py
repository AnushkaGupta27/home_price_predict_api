from flask import Flask,request,jsonify
import numpy as np
import pickle
model=pickle.load(open('regressor.pkl','rb'))
scaler_model=pickle.load(open('scaler.pkl','rb'))
app=Flask(__name__)
@app.route('/')
def home():
    return "heya heya"
@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        MedInc=float(request.form.get('MedInc'))
        HouseAge = float(request.form.get('HouseAge'))
        AveRooms = float(request.form.get('AveRooms'))
        AveBedrms = float(request.form.get('AveBedrms'))
        Population = float(request.form.get('Population'))
        AveOccup = float(request.form.get('AveOccup'))
        Latitude = float(request.form.get('Latitude'))
        Longitude = float(request.form.get('Longitude'))
        inp_query=np.array([[MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude]])
        new_data=scaler_model.transform(inp_query)
        result= model.predict(new_data)[0]
    return jsonify({'price':str(result)})
    #return "heyyoo"

if __name__ == '__main__':
    app.run(host="0.0.0.0")