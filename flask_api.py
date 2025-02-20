from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import json
import os

app = Flask(__name__)

# Маппинг категорий на файлы TFLite‑моделей
category_to_model_file = {
    "Сұлулық және денсаулық": "Beauty_and_Health_timeseries.tflite",
    "Киім": "Clothing_timeseries.tflite",
    "Балаларға арналған тауарлар": "Kids_timeseries.tflite",
    "Аксессуарлар": "Accessories_timeseries.tflite",
    "Үй және дача тауарлары": "Home_and_Garden_timeseries.tflite", 
    "Әшекейлер": "Jewelry_timeseries.tflite",
    "Дәріхана": "Pharmacy_timeseries.tflite",
    "Телефондар және гаджеттер": "Phones_and_Gadgets_timeseries.tflite",
    "Автокөлікке арналған тауарлар": "Automotive_timeseries.tflite",
    "Жиһаз": "Furniture_timeseries.tflite",
    "Демалыс және кітаптар": "Leisure_and_Books_timeseries.tflite",
    "Аяқ киім": "Footwear_timeseries.tflite",
    "Құрылыс және жөндеу": "Construction_timeseries.tflite",
    "Сыйлықтар, мерекелік тауарлар": "Gifts_and_Party_timeseries.tflite",
    "Компьютерлер": "Computers_timeseries.tflite",
    "Спорт және туризм": "Sports_and_Tourism_timeseries.tflite",
    "Канселярлық тауарлар": "Stationery_timeseries.tflite",
    "Үй техникасы": "Appliances_timeseries.tflite",
    "Азық-түлік": "Food_timeseries.tflite",
    "Теледидар, аудио, видео": "TV_Audio_Video_timeseries.tflite",
    "Жануарлар үшін тауарлар": "Pet_Supplies_timeseries.tflite",
}

# Маппинг категорий на файлы scaler‑параметров
category_to_scaler_file = {
    "Сұлулық және денсаулық": "Beauty_and_Health_timeseries_scalers.json",
    "Киім": "Clothing_timeseries_scalers.json",
    "Балаларға арналған тауарлар": "Kids_timeseries_scalers.json",
    "Аксессуарлар": "Accessories_timeseries_scalers.json",
    "Үй және дача тауарлары": "Home_and_Garden_timeseries_scalers.json",
    "Әшекейлер": "Jewelry_timeseries_scalers.json",
    "Дәріхана": "Pharmacy_timeseries_scalesr.json",
    "Телефондар және гаджеттер": "Phones_and_Gadgets_timeseries_scalers.json",
    "Автокөлікке арналған тауарлар": "Automotive_timeseries_scalers.json",
    "Жиһаз": "Furniture_timeseries_scalers.json",
    "Демалыс және кітаптар": "Leisure_and_Books_timeseries_scalers.json",
    "Аяқ киім": "Footwear_timeseries_scalers.json",
    "Құрылыс және жөндеу": "Construction_timeseries_scalers.json",
    "Сыйлықтар, мерекелік тауарлар": "Gifts_and_Party_timeseries_scalers.json",
    "Компьютерлер": "Computers_timeseries_scalers.json",
    "Спорт және туризм": "Sports_and_Tourism_timeseries_scalers.json",
    "Канселярлық тауарлар": "Stationery_timeseries_scalers.json",
    "Үй техникасы": "Appliances_timeseries_scalers.json",
    "Азық-түлік": "Food_timeseries_scalers.json",
    "Теледидар, аудио, видео": "TV_Audio_Video_timeseries_scalers.json",
    "Жануарлар үшін тауарлар": "Pet_Supplies_timeseries_scalers.json",
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)


