import numpy as np
import keras_cv

def transform_images(image):
    image = np.array(image)
    h, w, c = image.shape
    batch = image.reshape(1, h, w, c)
    inference_resizing = keras_cv.layers.Resizing(
        640, 640, pad_to_aspect_ratio=True, bounding_box_format="xywh"
    )
    image = inference_resizing(batch)
    print(image.numpy())
    return image