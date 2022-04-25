from flask import Flask, render_template, request
import pickle
import numpy as np
app=Flask(__name__)
smartphone_model=pickle.load(open('smartphone_model.pkl','rb'))
@app.route('/')
def smartphone_home():
    return render_template('smartphone_home.html')
@app.route('/predict',methods=['POST'])

def predict():
    RM= float(request.values['RM'])
    BP= float(request.values['BP'])
    MT= float(request.values['MT'])
    SW= float(request.values['SW'])
    SH= float(request.values['SH'])
    IM= float(request.values['IM'])
    CS= float(request.values['CS'])
    TT= float(request.values['TT'])
    PW= float(request.values['PW'])
    PH= float(request.values['PH'])
    input=np.array([BP,CS,IM,MT,PH,PW,RM,SH,SW,TT])
    input=np.reshape(input,(1,input.size))
    output=smartphone_model.predict(input)
    print(output)
    for x in output:
        if (x==3):
            output='Price range : Very high cost and Rank : 1'
        elif (x==2):
            output='Price range : High cost and Rank : 2'
        elif (x==1): 
            output='Price range : Medium cost and Rank : 3'
        elif (x==0):
            output='Price range : Low cost and Rank : 0'
    return render_template ('smartphone_result.html',prediction_text=" {} ".format(output))
if __name__=='__main__':
    app.run(port=8000)