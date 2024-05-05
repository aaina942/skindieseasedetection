from flask import Flask, render_template, request
import cv2
import numpy as np 
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)   

model = load_model('resnet.h5')
le = LabelEncoder()
label_names = ['Eczema', 'Warts Molluscum and other Viral Infections', 'Melanoma', 'Atopic Dermatitis', 'Basal Cell Carcinoma', 'Melanocytic Nevi', 'Benign Keratosis-like Lesions', 'Psoriasis pictures Lichen Planus ', 'Seborrheic Keratoses and other Benign Tumors ', 'Tinea Ringworm Candidiasis and other Fungal Infections']

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'imagefile' not in request.files:
            return render_template('index.html', error='No file selected')
        
        imagefile = request.files['imagefile']
        if imagefile.filename == '':
            return render_template('index.html', error='No file selected')
        
        image_path = "./images/" + imagefile.filename
        imagefile.save(image_path)

        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        image = preprocess_input(np.array([image]))
        predictions = model.predict(image)
        predicted_class_index = np.argmax(predictions)

        return render_template('index.html', predictions=label_names[predicted_class_index])

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
