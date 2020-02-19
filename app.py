from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

with open('./lib/models/LogisticRegression.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return(render_template('main.html'))

    if request.method == 'POST':
        alcohol                         =  request.form['alcohol']
        malic_acid                      =  request.form['malic_acid']
        ash                             =  request.form['ash']
        alcalinity_of_ash               =  request.form['alcalinity_of_ash']
        magnesium                       =  request.form['magnesium']
        total_phenols                   =  request.form['total_phenols']
        flavanoids                      =  request.form['flavanoids']
        nonflavanoid_phenols            =  request.form['nonflavanoid_phenols']
        proanthocyanins                 =  request.form['proanthocyanins']
        color_intensity                 =  request.form['color_intensity']
        hue                             =  request.form['hue']
        od280_od315_of_diluted_wines    =  request.form['od280_od315_of_diluted_wines']
        proline                         =  request.form['proline']

        input_variables = pd.DataFrame([[alcohol, malic_acid, ash, alcalinity_of_ash, magnesium,
                                           total_phenols, flavanoids, nonflavanoid_phenols,
                                           proanthocyanins, color_intensity, hue,
                                           od280_od315_of_diluted_wines, proline]],
                                       columns=['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                                               'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
                                               'proanthocyanins', 'color_intensity', 'hue',
                                               'od280/od315_of_diluted_wines', 'proline'],
                                       dtype=float)



        prediction = model.predict(input_variables)[0]
        if prediction == 0.0:
            prediction = "Poor Wine Quality"
        elif prediction == 1.0:
            prediction = "Good Wine Quality"
        else:
            prediction = "Fine Wine Quality"

        return render_template('main.html', result=prediction)



if __name__ == '__main__':
    app.run()

