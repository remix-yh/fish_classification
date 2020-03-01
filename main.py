import io
import base64
from PIL import Image
import numpy as np
import cv2
from flask import Flask, jsonify, render_template, request, redirect, url_for, render_template, flash, json
from fish import dl

app = Flask(__name__)

model = dl.initialize('./fish/model/model_weight.h5')

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/api/fish-classification', methods=['POST'])
def classification():
    img_decoded = base64.b64decode(request.json)
    img_binarystream = io.BytesIO(img_decoded)
    img_pil = Image.open(img_binarystream)
    p = dl.predict(model, img_pil)

    result = []
    haze = {}
    haze['name'] = 'ハゼ'
    haze['prediction'] = "{0:.3f}".format(p[0][0])

    kasago = {}
    kasago['name'] = 'カサゴ'
    kasago['prediction'] = "{0:.3f}".format(p[0][1])

    aji = {}
    aji['name'] = 'アジ'
    aji['prediction'] = "{0:.3f}".format(p[0][2])

    result.append(haze)
    result.append(kasago)
    result.append(aji)

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run()