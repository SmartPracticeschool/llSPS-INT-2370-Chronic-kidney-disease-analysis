import flask
import pandas as pd
import pickle


model = pickle.load(open('model/model.pkl', 'rb'))


app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return (flask.render_template('index.html'))

    if flask.request.method == 'POST':
        ID=flask.request.form['id']
        age = flask.request.form['age']
        bp = flask.request.form['bp']
        sg = flask.request.form['sg']
        al = flask.request.form['al']
        su = flask.request.form['su']
        rbc = flask.request.form['rbc']
        pc = flask.request.form['pc']
        pcc = flask.request.form['pcc']
        ba = flask.request.form['ba']
        bgr = flask.request.form['bgr']
        bu = flask.request.form['bu']
        sc = flask.request.form['sc']
        sod = flask.request.form['sod']
        pot = flask.request.form['pot']
        hemo = flask.request.form['hemo']
        pcv = flask.request.form['pcv']
        wc = flask.request.form['wc']
        rc = flask.request.form['rc']
        htn = flask.request.form['htn']
        dm = flask.request.form['dm']
        cad = flask.request.form['cad']
        appet = flask.request.form['appet']
        pe = flask.request.form['pe']
        ane = flask.request.form['ane']

        input_variables = pd.DataFrame([[ID,age, bp, sg, al, su, rbc, pc, pcc, ba,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc,htn,dm,cad,appet,pe,ane]],
                                       columns=['id','age', 'bp', 'sg', 'al', 'su',
                                                'rbc', 'pc', 'pcc', 'ba','bgr', 'bu','sc','sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane'],
                                       dtype='float',
                                       index=['input'])

        predictions = model.predict(input_variables)[0]
        print(predictions)

        return flask.render_template('index.html', original_input={'id':ID,'age':age, 'bp':bp, 'sg':sg, 'al':al, 'su':su,
                                                'rbc':rbc, 'pc':pc, 'pcc':pcc, 'ba':ba,'bgr':bgr, 'bu':bu,'sc':sc,'sod':sod,'pot':pot,'hemo':hemo,'pcv':pcv,'wc':wc,'rc':rc,'htn':htn,'dm':dm,'cad':cad,'appet':appet,'pe':pe,'ane':ane},
                                     result=predictions)


if __name__ == '__main__':
    app.run(debug=True)