import re
import cv2
import numpy as np
import json
import cvzone
from microdot import Microdot
import tensorflow.lite as tflite
from PIL import Image
from picamera2 import Picamera2
from libcamera import controls
# Import libcamera's controls class for autofocus and controlling the Camera Module 3
picam2 = Picamera2()
picam2.preview_configuration.main.size = (800,600)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
# Set the AfMode (Autofocus Mode) to be continuous
picam2.start()
CAMERA_WIDTH = 800
CAMERA_HEIGHT = 600
model_path='efficientdet_lite0.tflite'
label_path='labels.txt'

app = Microdot()

def load_labels(label_path):
    r"""Returns a list of labels"""
    with open(label_path) as f:
        labels = {}
        for line in f.readlines():
            m = re.match(r"(\d+)\s+(\w+)", line.strip())
            labels[int(m.group(1))] = m.group(2)
        return labels


def load_model(model_path):
    r"""Load TFLite model, returns a Interpreter instance."""
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def process_image(interpreter, image, input_index):
    r"""Process an image, Return a list of detected class ids and positions"""
    input_data = np.expand_dims(image, axis=0)  # expand to 4-dim

    # Process
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    # Get outputs
    output_details = interpreter.get_output_details()
    positions = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
    classes = np.squeeze(interpreter.get_tensor(output_details[1]['index']))
    scores = np.squeeze(interpreter.get_tensor(output_details[2]['index']))

    result = []

    for idx, score in enumerate(scores):
        if score > 0.5: # changed back to 0.5 to eliminate false positives
            result.append({'pos': positions[idx], 'id': classes[idx]})
    return result

def display_result(result, frame, labels):
    labels_with_counts = {item: 0 for item in labels.values()}
    lijst = []
    object_counts = {}
    for obj in result:
        id = obj['id']
        d=labels[id]
        lijst.append({'Object': d})
    for item in lijst:
        object_name = item['Object']
        if object_name in object_counts:
            object_counts[object_name] += 1
        else:
           object_counts[object_name] = 1
    for key, value in object_counts.items():
        labels_with_counts[key] += value
    return (labels_with_counts)

if __name__ == "__main__":

    model_path = 'efficientdet_lite0.tflite'
    label_path = 'labels.txt'
    interpreter = load_model(model_path)
    labels = load_labels(label_path)

    input_details = interpreter.get_input_details()

    # Get Width and Height
    input_shape = input_details[0]['shape']
    height = input_shape[1]
    width = input_shape[2]

    # Get input index
    input_index = input_details[0]['index']
    
    @app.get('/api/test')
    def get_test(request):
        msg = {"test":"oke"}
        print (msg)
        response = json.dumps(msg)
        return response, {'Content-Type': 'application/json'}
    
    @app.route('/api/id')
    def index(request):
        msg = {"identifier": "TensorFlow"}
        print (msg)
        response = json.dumps(msg)
        return response, {'Content-Type': 'application/json'}  
    
    @app.route('api/tsf')
    def index(request):
        im= picam2.capture_array()
        im=cv2.flip(im,-1)
        image = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        image = image.resize((width, height))
        top_result = process_image(interpreter, image, input_index)
        labels_with_counts = display_result(top_result, im, labels)
        return json.dumps(labels_with_counts)
    
    app.run()


