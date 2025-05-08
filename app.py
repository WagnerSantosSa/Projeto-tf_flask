from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limite de 16MB para uploads

# Cria a pasta de uploads se não existir
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# ====== NOVO BLOCO PARA BAIXAR O MODELO DO GOOGLE DRIVE ======
model_url = 'https://drive.google.com/uc?export=download&id=1meZ3bTrVSFs8PxcdGXw115Ac87DTVWYj'
model_path = 'modelo_treinado.h5'

if not os.path.exists(model_path):
    print('Baixando o modelo com gdown...')
    gdown.download(model_url, model_path, quiet=False)
    print('Download concluído!')
else:
    print('Modelo já existe localmente.')

# ====== CARREGAMENTO DO MODELO ======
model = tf.keras.models.load_model(model_path)
CLASSES = ['Sem Doença', 'Com Doença']

# Processamento de imagem
def process_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((128, 128))  # ajuste se necessário
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

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    print(f"Imagem recebida e salva em: {filepath}")

    image = process_image(filepath)
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
    return jsonify({'response': response, 'image_url': f"/{filepath}"})

if __name__ == '__main__':
    app.run()
