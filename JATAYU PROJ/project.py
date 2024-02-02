from flask import Flask, render_template, request
import lightgbm as lgb
import random
import pandas as pd


app = Flask(__name__)


model = lgb.Booster(model_file='./static/model.txt')
def make_prediction(model, df):
    pred = model.predict(df)
    return pred
    



@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/service')
def service():
    return render_template('service.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')
    
@app.route('/usecases')
def usecases():
    return render_template('usecases.html')


@app.route("/predict")
def view():
    with open('./static/state.txt') as f:
        states = f.read().splitlines()
    with open('./static/ClmAdmitDiagnosisCode.txt', 'r') as f:
        clmAdmitDiagnosisCodes = f.read().splitlines()
    with open('./static/ClmDiagnosisCode_1.txt', 'r') as f:
        ClmDiagnosisCode_1 = f.read().splitlines()
    with open('./static/ClmDiagnosisCode_2.txt', 'r') as f:
        ClmDiagnosisCode_2 = f.read().splitlines()
    with open('./static/ClmDiagnosisCode_3.txt', 'r') as f:
        ClmDiagnosisCode_3 = f.read().splitlines()
    with open('./static/ClmDiagnosisCode_4.txt', 'r') as f:
        ClmDiagnosisCode_4 = f.read().splitlines()
    return render_template('predictions.html', states=states, clmAdmitDiagnosisCodes=clmAdmitDiagnosisCodes, ClmDiagnosisCode_1=ClmDiagnosisCode_1, ClmDiagnosisCode_2=ClmDiagnosisCode_2, ClmDiagnosisCode_3=ClmDiagnosisCode_3, ClmDiagnosisCode_4=ClmDiagnosisCode_4)

@app.route("/predict_fraud", methods=['POST'])
def predict():
    if request.method == "POST":
        state = int(request.form['state'])
        age = int(request.form['age'])
        dob_year = int(request.form['dob_year'])
        dob_month = int(request.form['dob_month'])
        tot_reimbursed_amt = int(request.form['tot_reimbursed_amt'])
        tot_deductible_amt = int(request.form['tot_deductible_amt'])
        inscclaimamtreimbursed = int(request.form['inscclaimamtreimbursed'])
        aphysician = int(request.form['aphysician'][3:])
        ophysician = int(request.form['ophysician'][3:])
        otphysician = int(request.form['otphysician'][3:])
        Chr_Cond_Count = int(request.form['Chr_Cond_Count'])
        tot_claim_amt = int(request.form['tot_claim_amt'])
        ClmDiagnosisCode_1 = int(request.form['ClmDiagnosisCode_1'])
        ClmDiagnosisCode_2 = int(request.form['ClmDiagnosisCode_2'])
        ClmDiagnosisCode_3 = int(request.form['ClmDiagnosisCode_3'])
        ClmDiagnosisCode_4 = int(request.form['ClmDiagnosisCode_4'])
        Claim_Start_Month = int(request.form['Claim_Start_Month'])
        clmAdmitDiagnosisCode = int(request.form['clmAdmitDiagnosisCode'])
        county = random.randint(1, 1000)
        DiagnosisGroupCode = int(request.form['DiagnosisGroupCode'])
        vals = {
            "State": [state],
            "County": [county],
            "Age": [age],
            "Tot_Reimbursed_Amt": [tot_reimbursed_amt],
            "Tot_Deductible_Amt": [tot_deductible_amt],
            "DOB_year": [dob_year],
            "DOB_month": [dob_month],
            "InscClaimAmtReimbursed": [inscclaimamtreimbursed],
            "AttendingPhysician": [aphysician],
            "OperatingPhysician": [ophysician],
            "OtherPhysician": [otphysician],
            "ClmAdmitDiagnosisCode": [clmAdmitDiagnosisCode],
            "DiagnosisGroupCode": [DiagnosisGroupCode],
            "ClmDiagnosisCode_1": [ClmDiagnosisCode_1],
            "ClmDiagnosisCode_2": [ClmDiagnosisCode_2],
            "ClmDiagnosisCode_3": [ClmDiagnosisCode_3],
            "ClmDiagnosisCode_4": [ClmDiagnosisCode_4],
            "Claim_Start_Month": [Claim_Start_Month],
            "Total_Claim_Amt": [tot_claim_amt],
            "Chr_Cond_Count": [Chr_Cond_Count]
        }
        df = pd.DataFrame(vals)
        pred = make_prediction(model, df)
        return render_template('result.html', fraud = 1 if pred[0] > 0.5 else 0, prediction_accuracy = round(pred[0], 2))

    

@app.route('/result')
def result():
    return render_template('result.html', fraud = 1)

if __name__ == '__main__':
    app.run(debug=True)
