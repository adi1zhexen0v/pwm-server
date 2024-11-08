from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load('./model1.pkl')

default_values = {
    "temperature": -3.8,
    "humidity": 77,
    "cloudiness": 97,
    "pressure": 1015
}

@app.route('/')
def hello_world():
    return 'Hello, World!'
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    has_person = data.get('hasPerson')
    if has_person is False:
        return jsonify({'pwm': 0})

    temperature = data.get('temperature', default_values["temperature"])
    humidity = data.get('humidity', default_values["humidity"])
    cloudiness = data.get('cloudiness', default_values["cloudiness"])
    pressure = data.get('pressure', default_values["pressure"])
    lightLevel = data.get('lightLevel')

    if lightLevel is None:
        return jsonify({"error": "lightLevel is required"}), 400

    features = [[temperature, humidity, cloudiness, pressure, lightLevel]]
    pwm_value = model.predict(features)[0]

    return jsonify({'pwm': int(pwm_value)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
