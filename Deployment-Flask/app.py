
from flask import Flask, render_template, jsonify,  request , app
import numpy as np
import pickle
# prediction function 

app = Flask(__name__, template_folder='templates')
loaded_model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')
def ValuePredictor(to_predict_list): 
    to_predict = np.array(to_predict_list).reshape(1, 24) 
    
    result = loaded_model.predict(to_predict) 
    return result[0] 
  
@app.route('/predict', methods = ['POST']) 
def predict(): 
    if request.method == 'POST': 
        to_predict_list = request.form.to_dict() 
        to_predict_list = list(to_predict_list.values()) 
        to_predict_list = list(map(int, to_predict_list)) 
        result = ValuePredictor(to_predict_list)         
        if int(result)== 0: 
            prediction ='Disease is not Chronic'
        else: 
            prediction ='Disease is Chronic'            
        return render_template("index.html", prediction = prediction) 
    
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = loaded_model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
if __name__ == "__main__":
    app.run(debug=True)