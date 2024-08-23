import keras_cv

def predict(image_batch, model):
  prediction_decoder = keras_cv.layers.MultiClassNonMaxSuppression(
    bounding_box_format="xywh",
    from_logits=True,
    iou_threshold=0.1,
    confidence_threshold=0.02,
  )
  model.prediction_decoder = prediction_decoder

  y_pred = model.predict(image_batch)
  return y_pred