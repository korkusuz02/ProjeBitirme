
from flask import Flask, request, render_template
import pickle
import os
import numpy as np

flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

#"/" rotası için bir görünüm fonksiyonu olan index() tanımlanır ve home.html şablonunu döndürür.
@flask_app.route("/")
def index():
    return render_template("home.html")

#"/predict" rotası için bir görünüm fonksiyonu olan predict() tanımlanır. Bu fonksiyon, bir HTTP POST isteği aldığında çalışır.
@flask_app.route("/predict", methods=["POST"])
def predict():
    #request.form.values() kullanarak formdan gelen veriler elde edilir ve bir listeye dönüştürülür.
    float_features = [str(x) for x in request.form.values()]
    #Bu liste, numpy modülü kullanarak bir özellik vektörüne dönüştürülür.
    features = [np.array(float_features)]
    prediction = model.predict(features)
    if prediction == 0:
        result = 'Kalıcı, Lütfen en yakın sağlık kuruluşuyla irtibata geçiniz.'
    else:
        result = 'Kalıcı Değil, endişe endecek bir durum söz konusu değildir.'
    return render_template("home.html", result_of_prediction="-----Tahmin sonucumuz= {}".format(result))

if __name__ == "__main__":
    os.environ.setdefault('FLASK_ENV', 'development')
    flask_app.run(debug=True)
