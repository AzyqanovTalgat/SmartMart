from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import json
import os

app = Flask(__name__)


# Маппинг категорий на файлы TFLite‑моделей (ключи — английские имена, как в Flutter)
category_to_model_file = {
    "Beauty_and_Health": "Beauty_and_Health_timeseries.tflite",
    "Clothing": "Clothing_timeseries.tflite",
    "Kids": "Kids_timeseries.tflite",
    "Accessories": "Accessories_timeseries.tflite",
    "Home_and_Garden": "Home_and_Garden_timeseries.tflite",
    "Jewelry": "Jewelry_timeseries.tflite",
    "Pharmacy": "Pharmacy_timeseries.tflite",
    "Phones_and_Gadgets": "Phones_and_Gadgets_timeseries.tflite",
    "Automotive": "Automotive_timeseries.tflite",
    "Furniture": "Furniture_timeseries.tflite",
    "Leisure_and_Books": "Leisure_and_Books_timeseries.tflite",
    "Footwear": "Footwear_timeseries.tflite",
    "Construction": "Construction_timeseries.tflite",
    "Gifts_and_Party": "Gifts_and_Party_timeseries.tflite",
    "Computers": "Computers_timeseries.tflite",
    "Sports_and_Tourism": "Sports_and_Tourism_timeseries.tflite",
    "Stationery": "Stationery_timeseries.tflite",
    "Appliances": "Appliances_timeseries.tflite",
    "Food": "Food_timeseries.tflite",
    "TV_Audio_Video": "TV_Audio_Video_timeseries.tflite",
    "Pet_Supplies": "Pet_Supplies_timeseries.tflite",
}

# Маппинг категорий на файлы scaler‑параметров (ключи — английские имена)
category_to_scaler_file = {
    "Beauty_and_Health": "Beauty_and_Health_timeseries_scalers.json",
    "Clothing": "Clothing_timeseries_scalers.json",
    "Kids": "Kids_timeseries_scalers.json",
    "Accessories": "Accessories_timeseries_scalers.json",
    "Home_and_Garden": "Home_and_Garden_timeseries_scalers.json",
    "Jewelry": "Jewelry_timeseries_scalers.json",
    "Pharmacy": "Pharmacy_timeseries_scalers.json",
    "Phones_and_Gadgets": "Phones_and_Gadgets_timeseries_scalers.json",
    "Automotive": "Automotive_timeseries_scalers.json",
    "Furniture": "Furniture_timeseries_scalers.json",
    "Leisure_and_Books": "Leisure_and_Books_timeseries_scalers.json",
    "Footwear": "Footwear_timeseries_scalers.json",
    "Construction": "Construction_timeseries_scalers.json",
    "Gifts_and_Party": "Gifts_and_Party_timeseries_scalers.json",
    "Computers": "Computers_timeseries_scalers.json",
    "Sports_and_Tourism": "Sports_and_Tourism_timeseries_scalers.json",
    "Stationery": "Stationery_timeseries_scalers.json",
    "Appliances": "Appliances_timeseries_scalers.json",
    "Food": "Food_timeseries_scalers.json",
    "TV_Audio_Video": "TV_Audio_Video_timeseries_scalers.json",
    "Pet_Supplies": "Pet_Supplies_timeseries_scalers.json",
}


# Функция загрузки scaler параметров для входных признаков (features)
def load_scaler_params(json_file_path):
    try:
        with open(json_file_path, "r") as f:
            params = json.load(f)
        return params["features"]
    except Exception as e:
        print(f"Ошибка при загрузке scaler параметров: {e}")
        return None

# Функция загрузки scaler параметров для целевой переменной (target)
def load_target_scaler_params(json_file_path):
    try:
        with open(json_file_path, "r") as f:
            params = json.load(f)
        return params["target"]
    except Exception as e:
        print(f"Ошибка при загрузке target JSON-файла {json_file_path}: {e}")
        return None

# Функция масштабирования входных признаков: 
# scaled = (value - data_min) * scale + min
def scale_input_features(raw_features, scaler):
    scaled = []
    for i, val in enumerate(raw_features):
        scaled_val = (val - scaler["data_min"][i]) * scaler["scale"][i] + scaler["min"][i]
        scaled.append(scaled_val)
    return scaled

# Функция обратного масштабирования для целевой переменной:
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
    # Ожидается JSON следующей структуры:
    # {
    #   "category": "Computers",
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

    # Загружаем scaler параметры с кэшированием
    scaler_file = category_to_scaler_file[category]
    if category in loaded_scalers:
        features_scaler = loaded_scalers[category]["features"]
        target_scaler = loaded_scalers[category]["target"]
    else:
        features_scaler = load_scaler_params(scaler_file)
        target_scaler = load_target_scaler_params(scaler_file)
        if features_scaler is None or target_scaler is None:
            return jsonify({"error": "Ошибка при загрузке scaler параметров"}), 500
        loaded_scalers[category] = {
            "features": features_scaler,
            "target": target_scaler
        }

    # Масштабирование входных признаков
    scaled_features = scale_input_features(raw_features, features_scaler)
    # Формирование входного тензора с формой [1, 1, num_features]
    input_data = np.array([scaled_features], dtype=np.float32).reshape(1, 1, len(raw_features))

    # Загружаем TFLite модель (с кэшированием)
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
    # Предполагаем, что выход имеет форму [1, 1]
    scaled_pred = output[0][0]
    prediction = inverse_scale_output(scaled_pred, target_scaler)

    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
