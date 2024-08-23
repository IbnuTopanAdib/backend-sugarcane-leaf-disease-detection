import time
from absl import app, logging
import cv2
import keras
import keras_cv
from keras_cv import visualization
from keras_cv import bounding_box
import numpy as np
import tensorflow as tf
from flask import Flask, request, Response, jsonify, send_from_directory, abort
import os
from transform_images import transform_images
from draw_outputs import draw_outputs
from prediction import predict
from matplotlib import pyplot as plt
from io import BytesIO

size = 640
models_path = './checkpoint.keras'
class_mapping = {0: 'diseases-3Yep', 1: 'healthy', 2: 'mosaic', 3: 'redrot', 4: 'rust'}
output_path = './detections/'
reloaded_model = tf.keras.models.load_model(models_path)

reloaded_model.compile(
    classification_loss='focal',
    box_loss="smoothl1",
    optimizer=tf.keras.optimizers.SGD(
    global_clipnorm=10.0,
),
    jit_compile = False
)

app = Flask(__name__)

@app.route('/detections', methods=['POST'])
def get_detections():
    raw_images = []
    images = request.files.getlist("images")
    image_names = []
    for image in images:
        image_name = image.filename
        image_names.append(image_name)
        image.save(os.path.join(os.getcwd(), image_name))
        img_raw = tf.image.decode_image(
            open(image_name, 'rb').read(), channels=3)
        raw_images.append(img_raw)
        
    num = 0
    
    response = []

    for j in range(len(raw_images)):
        responses = []
        raw_img = raw_images[j]
        num+=1
        img = transform_images(raw_img)

        t1 = time.time()
        prediction_result = predict(img, reloaded_model)
        boxes = prediction_result['boxes']
        scores = prediction_result['confidence']
        classes = prediction_result['classes']
        nums = prediction_result['num_detections']
        t2 = time.time()
        print('time: {}'.format(t2 - t1))

        print('detections:')
        for i in range(nums[0]):
            print('\t{}, {}, {}'.format(class_mapping[int(classes[0][i])],
                                            np.array(scores[0][i]),
                                            np.array(boxes[0][i])))
            responses.append({
                "class": class_mapping[int(classes[0][i])],
                "confidence": float("{0:.2f}".format(np.array(scores[0][i])*100))
            })
        response.append({
            "image": image_names[j],
            "detections": responses
        })
        img = cv2.cvtColor(raw_img.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_mapping)
        cv2.imwrite(output_path + 'detection' + str(num) + '.jpg', img)
        print('output saved to: {}'.format(output_path + 'detection' + str(num) + '.jpg'))

    for name in image_names:
        os.remove(name)
    try:
        return jsonify({"response":response}), 200
    except FileNotFoundError:
        abort(404)

@app.route('/image', methods=['POST'])
def get_image():
    try:
        image = request.files["image"]
        image_name = image.filename
        
        image.save(os.path.join(os.getcwd(), image_name))
        
        img_raw = tf.image.decode_image(
            open(image_name, 'rb').read(), channels=3
        )
        
        image_ready = np.array(img_raw) / 255.0
        h, w, c = image_ready.shape
        batch = image_ready.reshape(1, h, w, c)

        inference_resizing = keras_cv.layers.Resizing(
            640, 640, pad_to_aspect_ratio=True, bounding_box_format="xywh"
        )

        image_batch = inference_resizing(batch)

        reloaded_model.prediction_decoder = keras_cv.layers.NonMaxSuppression(
            bounding_box_format="xywh",
            from_logits=True,
            iou_threshold=0.5,
            confidence_threshold=0.5,
        )

        y_pred = reloaded_model.predict(image_batch)
    
        visualization.plot_bounding_box_gallery(
            image_batch * 255,
            value_range=(0, 255),
            rows=0,
            cols=0,
            y_pred=y_pred,
            scale=5,
            font_scale=0.7,
            bounding_box_format="xywh",
            class_mapping=class_mapping,
        )
        
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        return Response(response=buf.getvalue(), status=200, mimetype='image/png')
    
    except FileNotFoundError:
        return Response(response="File not found", status=404)
if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port=5000)