from flask import Flask, request, render_template, jsonify
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Define reasonable limits for crop conditions and state mismatches
EXTREME_CONDITIONS = {
    "temperature": {"wheat": (0, 40), "rice": (15, 45), "soyabean": (10, 35)},
    "rainfall": {"rice": (1200, 4000), "wheat": (300, 1500)},
    "pH": {"wheat": (6.0, 7.5), "rice": (5.5, 7.5), "soyabean": (6.0, 7.0)},
    "N": {"wheat": (50, 200), "rice": (60, 220), "soyabean": (20, 80)},
    "P": {"wheat": (20, 60), "rice": (30, 80), "soyabean": (10, 50)},
    "K": {"wheat": (30, 120), "rice": (40, 150), "soyabean": (20, 90)}
}

# Updated STATE_CROP_MISMATCHES with recommended crops for each state
STATE_RECOMMENDED_CROPS = {
    "rajasthan": {
        "mismatches": ["rice", "black pepper", "coconut", "banana", "rubber", "tea", "coffee", "pineapple", "arecanut", "cardamom", "apple"],
        "recommended": ["wheat", "pearl millet", "soyabean", "mustard", "cumin", "groundnut"]
    },
    "tamil nadu": {
        "mismatches": ["apple", "wheat", "barley", "cherry", "saffron", "almond", "walnut", "apricot", "pear", "grapes"],
        "recommended": ["rice", "groundnut", "sugarcane", "cotton", "banana", "mango"]
    },
    "kerala": {
        "mismatches": ["wheat", "barley", "maize", "saffron", "apple", "cherry", "grapes", "almond", "apricot", "pear"],
        "recommended": ["rice", "coconut", "banana", "cassava", "black pepper", "rubber"]
    },
    "punjab": {
        "mismatches": ["coconut", "banana", "rubber", "tea", "coffee", "pineapple", "arecanut", "cardamom", "mango", "papaya"],
        "recommended": ["wheat", "rice", "cotton", "sugarcane", "maize"]
    },
    "gujarat": {
        "mismatches": ["apple", "black pepper", "cardamom", "tea", "coffee", "arecanut", "saffron"],
        "recommended": ["cotton", "groundnut", "castor", "pearl millet", "sorghum"]
    },
    "madhya pradesh": {
        "mismatches": ["coconut", "tea", "coffee", "black pepper", "rubber"],
        "recommended": ["soyabean", "wheat", "maize", "cotton", "gram"]
    },
    "uttar pradesh": {
        "mismatches": ["coconut", "banana", "rubber", "tea", "coffee", "arecanut", "black pepper"],
        "recommended": ["wheat", "rice", "sugarcane", "potato", "mustard"]
    },
    "west bengal": {
        "mismatches": ["apple", "walnut", "almond", "saffron"],
        "recommended": ["rice", "jute", "potato", "wheat", "mustard"]
    },
    "karnataka": {
        "mismatches": ["apple", "walnut", "cherry", "saffron"],
        "recommended": ["rice", "ragi", "sugarcane", "cotton", "groundnut"]
    },
    "bihar": {
        "mismatches": ["coconut", "tea", "coffee", "rubber", "black pepper", "arecanut"],
        "recommended": ["wheat", "rice", "maize", "potato", "sugarcane"]
    }
}

STATE_SOIL_NUTRIENTS = {
    "rajasthan": {"pH": (7.0, 9.0), "N": (20, 100), "P": (10, 40), "K": (50, 150)},
    "tamil nadu": {"pH": (5.5, 7.5), "N": (30, 150), "P": (15, 50), "K": (60, 180)},
    "kerala": {"pH": (4.5, 6.5), "N": (40, 120), "P": (20, 60), "K": (70, 200)},
    "punjab": {"pH": (6.0, 8.5), "N": (50, 180), "P": (25, 70), "K": (80, 220)},
    "gujarat": {"pH": (6.5, 8.5), "N": (30, 140), "P": (20, 55), "K": (60, 170)},
    "madhya pradesh": {"pH": (6.0, 8.0), "N": (40, 160), "P": (15, 60), "K": (70, 190)},
    "uttar pradesh": {"pH": (6.5, 8.0), "N": (50, 170), "P": (20, 65), "K": (75, 210)},
    "west bengal": {"pH": (5.0, 7.5), "N": (40, 130), "P": (20, 55), "K": (65, 180)},
    "karnataka": {"pH": (5.5, 7.5), "N": (35, 140), "P": (18, 60), "K": (70, 190)},
    "bihar": {"pH": (6.0, 8.0), "N": (45, 150), "P": (20, 60), "K": (75, 200)}
}

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST', "GET"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    else:
        errors = []
        warnings = []
        soil_warnings = []
        crop_recommendations = []
        
        try:
            data = CustomData(
                N=float(request.form.get('N')),
                P=float(request.form.get('P')),
                K=float(request.form.get("K")),
                pH=float(request.form.get("pH")),
                rainfall=float(request.form.get("rainfall")),
                temperature=float(request.form.get("temperature")),
                Area_in_hectares=float(request.form.get('Area_in_hectares')),
                State_Name=request.form.get("State_Name").lower(),
                Crop_Type=request.form.get("Crop_Type"),
                Crop=request.form.get("Crop").lower()
            )
            
            # Check for negative or zero pH value
            if data.pH < 0:
                errors.append("pH value cannot be negative.")
            
            # Check for negative or zero rainfall value
            if data.rainfall <= 0:
                errors.append("Rainfall value cannot be negative or zero.")
            
            # Check for negative or zero N, P, K values
            if data.N <= 0:
                errors.append("Nitrogen (N) value cannot be negative or zero.")
            if data.P <= 0:
                errors.append("Phosphorus (P) value cannot be negative or zero.")
            if data.K <= 0:
                errors.append("Potassium (K) value cannot be negative or zero.")

            # Check state-specific soil conditions
            if data.State_Name in STATE_SOIL_NUTRIENTS:
                for nutrient, limits in STATE_SOIL_NUTRIENTS[data.State_Name].items():
                    min_val, max_val = limits
                    value = getattr(data, nutrient)
                    if not (min_val <= value <= max_val):
                        soil_warnings.append(f"{nutrient.upper()} levels in {data.State_Name.capitalize()} usually range between {min_val}-{max_val}. Entered: {value}")

            # Check if the selected crop is not suitable for the entered state
            if data.State_Name in STATE_RECOMMENDED_CROPS:
                if data.Crop in STATE_RECOMMENDED_CROPS[data.State_Name]["mismatches"]:
                    warnings.append(f"{data.Crop.capitalize()} is not suitable for {data.State_Name.capitalize()}.")
                    
                    # Add crop recommendations
                    recommended_crops = STATE_RECOMMENDED_CROPS[data.State_Name]["recommended"]
                    crop_recommendations.append(f"Recommended crops for {data.State_Name.capitalize()}: {', '.join(crop.capitalize() for crop in recommended_crops)}")

            # Check temperature and rainfall suitability
            if data.Crop in EXTREME_CONDITIONS["temperature"]:
                min_temp, max_temp = EXTREME_CONDITIONS["temperature"][data.Crop]
                if not (min_temp <= data.temperature <= max_temp):
                    warnings.append(f"Temperature too {'low' if data.temperature < min_temp else 'high'} for {data.Crop.capitalize()}.")

            if data.Crop in EXTREME_CONDITIONS["rainfall"]:
                min_rain, max_rain = EXTREME_CONDITIONS["rainfall"][data.Crop]
                if not (min_rain <= data.rainfall <= max_rain):
                    warnings.append(f"Rainfall too {'low' if data.rainfall < min_rain else 'high'} for {data.Crop.capitalize()}.")

            if errors:
                return render_template("index.html", errors=errors)
            
            new_data = data.get_data_as_dataframe()
            predict_pipeline = PredictPipeline()
            pred = predict_pipeline.predict(new_data)

            production = round(pred[0], 2)
            yield_value = round(production / data.Area_in_hectares, 2)

            final_result = f"Predicted Crop Production: {production} tons"
            yield_result = f"Predicted Yield: {yield_value} tons/hectare"
            
            return render_template("index.html", 
                                   final_result=final_result, 
                                   yield_result=yield_result, 
                                   warnings=warnings, 
                                   soil_warnings=soil_warnings,
                                   crop_recommendations=crop_recommendations)
        
        except ValueError as e:
            return render_template("index.html", errors=["Invalid input detected. Please enter valid numbers for all fields."])

if __name__ == "__main__":
    app.run(debug=True)