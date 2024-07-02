import pickle
from flask import Flask, render_template, request
app = Flask(__name__)
model=pickle.load(open('iris_model.sav','rb'))
label_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
@app.route('/')
def home():
    return render_template('index.html',result='')
@app.route('/predict',methods=['POST','GET'])
def predict():
    sepal_length=float(request.form['SepalLengthCm'])
    sepal_width=float(request.form['SepalWidthCm'])
    petal_length=float(request.form['PetalLengthCm'])
    petal_width=float(request.form['PetalWidthCm'])
    result=label_mapping[model.predict([[sepal_length,sepal_width,petal_length,petal_width]])[0]]
    return render_template('index.html',result=result)

if __name__ == '__main__':
    app.run(debug=True)