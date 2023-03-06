import os
from flask import Flask, render_template, request, send_from_directory
import static.models.unet_xception as mux
import static.models.deeplab_v3plus as mdv3
import static.models.cityscapes as cityscapes
import tensorflow as tf

import numpy as np
from PIL import Image


app = Flask(__name__)

STATIC_FOLDER = 'static/'
MODEL_FOLDER = STATIC_FOLDER + 'models/'
UPLOAD_FOLDER = STATIC_FOLDER + 'uploads/'
RESULT_FOLDER = STATIC_FOLDER + 'results/'

@app.before_first_request
def load__model():
    """
    Load model
    :return: model (global variable)
    """
    print('[INFO] Model Loading ........')
    global model, resize
    
    #backbone="mobilenetv2" #"xception"
    model_name = "deeplab_v3plus_512_augment"

    resize = int(model_name.replace("_augment", "").split("_")[-1])
    model = mdv3.get_model(
                    weights="cityscapes",
                    input_tensor=None,
                    input_shape=(resize, resize, 3),
                    classes=8,
                    backbone="mobilenetv2", #"xception",
                    OS=36,
                    alpha=1.0,
                    activation="softmax",
                    model_name=model_name,
                )
    print(model.name)

    model.load_weights(MODEL_FOLDER + model_name + ".h5")
    

def predict_(fullpath_image):

    #input_img = image.load_img(fullpath, target_size=(150, 150, 3))
    input_img = Image.open(fullpath_image).resize((resize, resize))
    
    # Prediction:
    result = Image.fromarray(
        cityscapes.cityscapes_category_ids_to_category_colors(
            np.squeeze(
                np.argmax(
                    model.predict(np.expand_dims(input_img, 0)),
                    axis=-1,
                )
            )
        )
    )
    #result.save(os.path.join(RESULT_FOLDER +"pred1.png"), format="PNG")

    return result #RESULT_FOLDER + "pred1.png"

# Home Page
@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        fullname = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(fullname)

        result = predict_(fullname)
        result.save(os.path.join(RESULT_FOLDER +"pred1.png"), format="PNG")

        return render_template('index.html', image_file_name=file.filename,
                               predict=True, result=RESULT_FOLDER +"pred1.png")
    else:

        return render_template('index.html', predict=False)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        fullname = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(fullname)

        result = predict_(fullname)
        result.save(os.path.join(RESULT_FOLDER +"pred1.png"), format="PNG")
        
        with open(os.path.join(RESULT_FOLDER +"pred1.png"),  "rb") as f :
            img_enc = base64.b64encode(f.read())

        return jsonify({"image":img_enc})


@app.route('/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/<filename>')
def result_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/<filename>')
def ground_truth_seg_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT',5000))
    app.run(host="0.0.0.0", port=port, debug=True)
