from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import json
import os

app = Flask(__name__)

# Маппинг категорий на файлы TFLite‑моделей
category_to_model_file = {
    "Сұлулық және денсаулық": "tflite_models/Beauty_and_Health_timeseries.tflite",
    "Киім": "tflite_models/Clothing_timeseries.tflite",
    "Балаларға арналған тауарлар": "tflite_models/Kids_timeseries.tflite",
    "Аксессуарлар": "tflite_models/Accessories_timeseries.tflite",
    "Үй және дача тауарлары": "tflite_models/Home_and_Garden_timeseries.tflite", 
    "Әшекейлер": "tflite_models/Jewelry_timeseries.tflite",
    "Дәріхана": "tflite_models/Pharmacy_timeseries.tflite",
    "Телефондар және гаджеттер": "tflite_models/Phones_and_Gadgets_timeseries.tflite",
    "Автокөлікке арналған тауарлар": "tflite_models/Automotive_timeseries.tflite",
    "Жиһаз": "tflite_models/Furniture_timeseries.tflite",
    "Демалыс және кітаптар": "tflite_models/Leisure_and_Books_timeseries.tflite",
    "Аяқ киім": "tflite_models/Footwear_timeseries.tflite",
    "Құрылыс және жөндеу": "tflite_models/Construction_timeseries.tflite",
    "Сыйлықтар, мерекелік тауарлар": "tflite_models/Gifts_and_Party_timeseries.tflite",
    "Компьютерлер": "tflite_models/Computers_timeseries.tflite",
    "Спорт және туризм": "tflite_models/Sports_and_Tourism_timeseries.tflite",
    "Канселярлық тауарлар": "tflite_models/Stationery_timeseries.tflite",
    "Үй техникасы": "tflite_models/Appliances_timeseries.tflite",
    "Азық-түлік": "tflite_models/Food_timeseries.tflite",
    "Теледидар, аудио, видео": "tflite_models/TV_Audio_Video_timeseries.tflite",
    "Жануарлар үшін тауарлар": "tflite_models/Pet_Supplies_timeseries.tflite",
}

# Маппинг категорий на файлы scaler‑параметров
category_to_scaler_file = {
    "Сұлулық және денсаулық": "models/scaler_params/Beauty_and_Health_timeseries_scalers.json",
    "Киім": "models/scaler_params/Clothing_timeseries_scalers.json",
    "Балаларға арналған тауарлар": "models/scaler_params/Kids_timeseries_scalers.json",
    "Аксессуарлар": "models/scaler_params/Accessories_timeseries_scalers.json",
    "Үй және дача тауарлары": "models/scaler_params/Home_and_Garden_timeseries_scalers.json",
    "Әшекейлер": "models/scaler_params/Jewelry_timeseries_scalers.json",
    "Дәріхана": "models/scaler_params/Pharmacy_timeseries_scalesr.json",
    "Телефондар және гаджеттер": "models/scaler_params/Phones_and_Gadgets_timeseries_scalers.json",
    "Автокөлікке арналған тауарлар": "models/scaler_params/Automotive_timeseries_scalers.json",
    "Жиһаз": "models/scaler_params/Furniture_timeseries_scalers.json",
    "Демалыс және кітаптар": "models/scaler_params/Leisure_and_Books_timeseries_scalers.json",
    "Аяқ киім": "models/scaler_params/Footwear_timeseries_scalers.json",
    "Құрылыс және жөндеу": "models/scaler_params/Construction_timeseries_scalers.json",
    "Сыйлықтар, мерекелік тауарлар": "models/scaler_params/Gifts_and_Party_timeseries_scalers.json",
    "Компьютерлер": "models/scaler_params/Computers_timeseries_scalers.json",
    "Спорт және туризм": "models/scaler_params/Sports_and_Tourism_timeseries_scalers.json",
    "Канселярлық тауарлар": "models/scaler_params/Stationery_timeseries_scalers.json",
    "Үй техникасы": "models/scaler_params/Appliances_timeseries_scalers.json",
    "Азық-түлік": "models/scaler_params/Food_timeseries_scalers.json",
    "Теледидар, аудио, видео": "models/scaler_params/TV_Audio_Video_timeseries_scalers.json",
    "Жануарлар үшін тауарлар": "models/scaler_params/Pet_Supplies_timeseries_scalers.json",
}

# Функция загрузки scaler параметров из JSON
def load_scaler_params(json_file_path):
    try:
        with open(json_file_path, "r") as f:
            params = json.load(f)
        # Предполагаем, что нам нужны параметры для "features"
        return params["features"]
    except Exception as e:
        print(f"Ошибка при загрузке scaler параметров: {e}")
        return None

# Функция масштабирования входных признаков:
# scaled = (value - data_min) * scale + min
def scale_input_features(raw_features, scaler):
    scaled = []
    for i, val in enumerate(raw_features):
        scaled_val = (val - scaler["data_min"][i]) * scaler["scale"][i] + scaler["min"][i]
        scaled.append(scaled_val)
    return scaled

# Функция обратного масштабирования для y:
# real = (scaled - min) / scale + data_min
def inverse_scale_output(scaled_val, scaler):
    return (scaled_val - scaler["min"][0]) / scaler["scale"][0] + scaler["data_min"][0]

# Функция загрузки TFLite модели
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Кэш для загруженных моделей и scaler параметров (для ускорения повторных запросов)
loaded_models = {}
loaded_scalers = {}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Ожидается JSON:
    # {
    #   "category": "Компьютерлер",
    #   "total_store_profit": ...,
    #   "avg_discount": ...,
    #   "avg_price": ...,
    #   "returns_count": ...,
    #   "population": ...
    # }
    category = data.get("category")
    if category is None:
        return jsonify({"error": "Не указана категория"}), 400
    if category not in category_to_model_file or category not in category_to_scaler_file:
        return jsonify({"error": f"Неверная категория: {category}"}), 400

    try:
        store_profit = float(data.get("total_store_profit", 0))
        avg_discount = float(data.get("avg_discount", 0))
        avg_price = float(data.get("avg_price", 0))
        returns_count = float(data.get("returns_count", 0))
        population = float(data.get("population", 0))
    except Exception as e:
        return jsonify({"error": "Неверный формат входных данных", "details": str(e)}), 400

    raw_features = [store_profit, avg_discount, avg_price, returns_count, population]

    # Загружаем scaler для выбранной категории (с кэшированием)
    scaler_file = category_to_scaler_file[category]
    if category in loaded_scalers:
        scaler = loaded_scalers[category]
    else:
        scaler = load_scaler_params(scaler_file)
        if scaler is None:
            return jsonify({"error": "Ошибка при загрузке scaler параметров"}), 500
        loaded_scalers[category] = scaler

    # Масштабируем входные признаки
    scaled_features = scale_input_features(raw_features, scaler)
    # Формируем входной тензор с формой [1, 1, num_features]
    input_data = np.array([scaled_features], dtype=np.float32).reshape(1, 1, len(raw_features))

    # Загружаем модель для выбранной категории (с кэшированием)
    model_file = category_to_model_file[category]
    if category in loaded_models:
        interpreter = loaded_models[category]
    else:
        if not os.path.exists(model_file):
            return jsonify({"error": f"Модель не найдена: {model_file}"}), 500
        interpreter = load_tflite_model(model_file)
        loaded_models[category] = interpreter

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    try:
        interpreter.invoke()
    except Exception as e:
        return jsonify({"error": "Ошибка при инференсе", "details": str(e)}), 500

    output = interpreter.get_tensor(output_details[0]['index'])
    # Предположим, что выход имеет форму [1, 1]
    scaled_pred = output[0][0]
    prediction = inverse_scale_output(scaled_pred, scaler)

    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)

