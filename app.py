import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  
import pickle

#procfile.txt 
#web: gunicorn app:app
#first file that we have to run first : flask server name
app = Flask(__name__, static_folder='./static')
CORS(app)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})



@app.route('/')
def home():
    # with css and html
    # render hello world
    return render_template('index.html')


@app.route('/predict/Banglore',methods=['POST'])
def predict_banglore():
    pkl_file = open('./Predictions/banglore/model.pkl','rb')
    bengalore_model = pickle.load(open('./Predictions/banglore/model.pkl', 'rb'))
    # index_dict = pickle.load(pkl_file)
# if request.method == 'POST': and location exits
    if request.method == 'POST':
        result = request.json

        index_dict = pickle.load(open('./Predictions/banglore/cat','rb'))
        location_cat = pickle.load(open('./Predictions/banglore/location_cat','rb'))

        new_vector = np.zeros(151)

        result_location = result['location']

        if result_location:
            if result_location not in location_cat:
                new_vector[146] = 1
            else:
                new_vector[index_dict[str(result['location'])]] = 1


            new_vector[index_dict[str(result['area'])]] = 1

            new_vector[0] = result['sqft']
            new_vector[1] = result['bath']
            new_vector[2] = result['balcony']
            new_vector[3] = result['size']

         
            square_fit = result['sqft']
            area = result['area']
            size = result['size']
            bathroom = result['bath']
            balcony = result['balcony']

            new = [new_vector]

            prediction = bengalore_model.predict(new)
            # print(prediction)

            return jsonify({'location': result_location,
                            'square_fit': square_fit,
                            'area': area,
                            'size': size,
                            'bathroom': bathroom,
                            'balcony': balcony,
                            'prediction': prediction[0]})
        
        else:
            return render_template('index.html', Predict_score ='Please enter the location')


@app.route('/predict/Delhi',methods=['POST'])
def predict_delhi():
    pkl_file = open('./Predictions/delhi/delhi_model.pkl','rb')
    delhi_model = pickle.load(open('./Predictions/delhi/delhi_model.pkl', 'rb'))
    index_dict = pickle.load(pkl_file)


    if request.method == 'POST':
        result = request.json
        # return jsonify(result)

        new_vector = np.zeros(151)

        result_location = result['location']

        index_dict = pickle.load(open('./Predictions/delhi/index_dict','rb'))
        location_cat = pickle.load(open('./Predictions/delhi/location_cat','rb'))
        # return jsonify(location_cat)
        if result_location:
            if result_location not in location_cat:
                new_vector[146] = 1
            else:
                new_vector[index_dict[str(result['location'])]] = 1


            # new_vector[index_dict[str(result['sqft'])]] = 1

            new_vector[0] = result['sqft']
            new_vector[1] = result['size']
            new_vector[2] = result['bath']

            area = result['sqft']
            size = result['size']
            bathroom = result['bath']

            new = [new_vector]

            prediction = delhi_model.predict(new)
            # print(prediction)

            return jsonify({'location': result_location,
                            'area': area,
                            'size': size,
                            'bathroom': bathroom,
                            'prediction': prediction[0]})
        
        else:
            return render_template('index.html', Predict_score ='Please enter the location')

if __name__ == "__main__":
    app.run(debug=True)
