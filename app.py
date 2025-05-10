from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image

try:
    import tflite_runtime.interpreter as tflite
    print("Usando tflite-runtime")
except ImportError:
    import tensorflow as tf
    tflite = tf.lite
    print("Usando tensorflow.lite")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

model_path = 'modelo_treinado.tflite'
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASSES = ['Sem Doença', 'Com Doença']

def process_image(image_stream):
    image = Image.open(image_stream).convert('RGB')
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0).astype(np.float32)
    return image_array

def predict_tflite(image_array):
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

def verificar_saudacao(mensagem):
    saudacoes = ['bom dia', 'boa tarde', 'boa noite', 'olá', 'ola', 'oi']
    for s in saudacoes:
        if s in mensagem.lower():
            return True
    return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/message', methods=['POST'])
def message():
    data = request.json
    user_message = data.get('message', '')
    response = ''

    if verificar_saudacao(user_message):
        response = "Olá, como posso ajudar você hoje?"
    elif 'imagem' in user_message.lower():
        response = "Claro, por favor envie a imagem para análise."
    else:
        response = f"Você disse: {user_message}"

    return jsonify({'response': response})

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'response': 'Nenhuma imagem enviada.'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'response': 'Nome de arquivo vazio.'})

    print(f"Imagem recebida (não salva no disco).")

    image = process_image(file.stream)
    prediction = predict_tflite(image)
    confidence = float(np.max(prediction)) * 100
    class_index = np.argmax(prediction)

    if class_index == 0 and confidence < 50:
        predicted_class = 'Com Doença'
        confidence = 100 - confidence
    else:
        predicted_class = CLASSES[class_index]

    print(f"Diagnóstico: {predicted_class} (Acurácia: {confidence:.2f}%)")

    response = f"Diagnóstico: {predicted_class} (Acurácia: {confidence:.2f}%)"
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run()
