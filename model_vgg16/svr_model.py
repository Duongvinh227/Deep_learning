
import pickle
from keras.models import load_model
import cv2
import tensorflow as tf
import numpy as np


# Load model VGG
tf_session = tf.compat.v1.Session()
tf_graph = tf.compat.v1.get_default_graph()

with tf_session.as_default():
    with tf_graph.as_default():
        model = load_model("model_vgg.h5")
        model.load_weights("model_vgg_weights.hdf5")
print(type(model))
# Load labels encoder
file = open('labels.pkl', 'rb')
encoder = pickle.load(file)
file.close()
cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_org = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5).copy()

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == key:# ord('r'):
        frame = cv2.resize(frame, dsize=(128, 128))
        frame = np.expand_dims(frame, axis=0)

        # Predict
        with tf_session.as_default():
            with tf_graph.as_default():
                result = model.predict(frame)[0]
                if len(result)>1:
                    result_id = np.argmax(result)
                    print(type(result_id))
                    if result[result_id] > 0.9:
                        ID = encoder.classes_[result_id]
                    else:
                        ID = "Unknown"
                else:
                    if result[0] > 0.8:
                        ID = encoder.classes_[1]
                    else:
                        ID = encoder.classes_[0]
                cv2.putText(frame_org, ID, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)



    cv2.imshow('frame', frame_org)

cap.release()
cv2.destroyAllWindows()
