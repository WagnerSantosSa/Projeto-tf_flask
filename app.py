from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limite de 16MB para uploads

# ====== CARREGAMENTO DO MODELO LOCAL ======
model_path = 'modelo_treinado.h5'
model = tf.keras.models.load_model(model_path)
CLASSES = ['Sem Doença', 'Com Doença']

# Processamento de imagem a partir de stream
def process_image(image_stream):
    image = Image.open(image_stream).convert('RGB')
    image = image.resize((128, 128))  # ajuste conforme seu treinamento
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Respostas de saudação
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
    prediction = model.predict(image)[0]
    confidence = float(np.max(prediction)) * 100
    class_index = np.argmax(prediction)

    # Lógica aprimorada: incerteza abaixo de 50% reverte o diagnóstico
    if class_index == 0 and confidence < 50:
        predicted_class = 'Com Doença'
        confidence = 100 - confidence  # reflete incerteza
    else:
        predicted_class = CLASSES[class_index]

    print(f"Diagnóstico: {predicted_class} (Acurácia: {confidence:.2f}%)")

    response = f"Diagnóstico: {predicted_class} (Acurácia: {confidence:.2f}%)"
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run()
