#
#   @Author: Simone Orlando
#   Through a convolutional neural network this program is able to recognize if the person, framed by the webcam, is wearing a protective mask or not.
#
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image


# Import and compile the convolutional neural network
json_file = open('./model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
loaded_model.load_weights("./model.h5")
loss = tf.keras.losses.CategoricalCrossentropy()
lr = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
metrics = ['accuracy']
loaded_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Start the webcam
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

# Convert the image from the webcam in the format accepted by the CNN
def prepare_image2 (img):
    cvt_image =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(cvt_image)
    im_resized = im_pil.resize((512, 512))
    img_array = tf.keras.preprocessing.image.img_to_array(im_resized)
    image_array_expanded = np.expand_dims(img_array, axis = 0)
    return tf.keras.applications.mobilenet.preprocess_input(image_array_expanded)

response = 2 # Inizialize the variable that will contain the prediction
while(True):
    ret, image = cap.read()
    img_array = prepare_image2(image)
    # Do the prediction for the current fram
    predictions = loaded_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class_int = np.argmax(predictions[0])
    response = predicted_class_int

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    thickness = 2
    # Add a label with the prediction to the current frame before showing it
    if (response == 0):
        color = (0, 0, 255)
        image = cv2.putText(image, 'Danger: no one wears a mask', org, font, fontScale, color, thickness, cv2.LINE_AA)
    elif (response==1):
        color = (0, 255, 0)
        image = cv2.putText(image, 'Safe: everyone wears a mask', org, font, fontScale, color, thickness, cv2.LINE_AA)
    else:
        color = (0, 255, 0)
        image = cv2.putText(image, 'Safe: everyone wears a mask', org, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
