from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
from src.functions import *
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "static/img"
IMAGE_SIZE = (20, 10)
detection_graph = init_graph()

@app.route('/')
def upload():
    return render_template('upload_image.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        return redirect(url_for('predict', image_path=f.filename))

@app.route('/predict/<image_path>')
def predict(image_path):
    image_np = display_pred(image_path, detection_graph)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
    plt.savefig('static/img/plot'+image_path)
    return render_template("predict.html",img_predict="img/plot"+image_path)


if __name__ == '__main__':
    app.run(debug=True)