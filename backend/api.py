from backend.utils.common import serialize_mongo_document
from fastapi import FastAPI, HTTPException, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, create_model
import joblib
from pathlib import Path
import pandas as pd
from typing import Dict, Type, Union, List
import yaml
from pymongo import MongoClient
import os 
from collections import Counter

with open("params.yaml") as f:
    params = yaml.safe_load(f)

numericas = params['continuas'] + params['discretas'] 
categoricas = params['categoricas']   

inputFile = 'data/top_features.csv'
df_features =  pd.read_csv(inputFile)

df =  pd.read_csv("data/results.csv")
best_model_name = df.iloc[0]["model"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL(s)
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],  # Specify allowed methods
    allow_headers=["Authorization", "Content-Type"],  # Specify allowed headers
)

model = joblib.load(f"models/{best_model_name}")
preprocessor = joblib.load('models/preprocessor.pkl')

UPLOAD_FOLDER = "input"
input_folder = Path(UPLOAD_FOLDER)
input_folder.mkdir(parents=True, exist_ok=True)

# Dynamically create a Pydantic model based on the CSV columns
def create_dynamic_model() -> Type:
    fields: Dict[str, Union[float, str]] = {col: (float, ...) for col in numericas}
    fields.update({col: (str, ...) for col in categoricas})
    print(fields)
    return create_model("PredictionRequest", **fields)

# Create the dynamic model
PredictionRequest = create_dynamic_model()

@app.post("/predict")
def predict(data:  List[PredictionRequest]):
    print(data)
    try: 
        predictions = []
        features = df_features['feature'].values
        for sample in data:
            data_dict = sample.dict()    
            input_data = pd.DataFrame([data_dict])
            X_transformed_df = preprocess_sample(input_data,features)
            
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_transformed_df)[0] 
                class_names = model.classes_  
                probability_dict = {str(class_name): float(prob) for class_name, prob in zip(class_names, proba)}               
            else:
                probability_dict = None  # If no probability support
            
            target_mapping = {
                'Pedido insuficiente': 0,
                'Posible producto eliminando de catalogo': 1,
                'Posible quiebre de stock por pedido insuficiente': 2,
                'Posible venta atípica': 3,
                'Producto sano': 4,
                'inventario negativo': 5,
                'producto nuevo sin movimiento': 6
            }
            reverse_target_mapping = {v: k for k, v in target_mapping.items()}
            probability_dict_with_labels = {
                reverse_target_mapping[int(class_name)]: prob 
                for class_name, prob in probability_dict.items()
            }
            predictions.append(probability_dict_with_labels) 

        response = {"predictions": predictions}
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    # Ensure the uploaded file is of the correct type
    if not file.filename.endswith(".xlsx"):
        return JSONResponse(content={"message": "Invalid file type. Only .xlsx files are allowed."}, status_code=400)
    
    # Create file path
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    
    # Save the file
    try:
        with open(file_path, "wb") as f:
            content = await file.read()  # Read the content of the file
            f.write(content)  # Write the content to the file

        response = JSONResponse(content={"message": "File uploaded successfully", "file_path": file_path})
        response.headers["Content-Type"] = "application/json"
        return response

    except Exception as e:
        return JSONResponse(content={"message": f"Failed to save file: {str(e)}"}, status_code=500)

@app.get("/results")
async def results_by_date(fecha: str = Query(..., description="Filter by 'Fecha' field")):
    try:
        client = MongoClient("mongodb://localhost:27017/") 
        db = client["retail_classificator"]
        collection = db["results"]

        
        print(fecha)
        documents = list(collection.find({"Fecha": fecha}))
        serialized_documents = [serialize_mongo_document(doc) for doc in documents]
        if not serialized_documents:
            raise HTTPException(status_code=404, detail="No documents found with the given Fecha.")
        

        fields_of_interest = [
            "Pedido insuficiente",
            "Posible producto eliminando de catalogo",
            "Posible quiebre de stock por pedido insuficiente",
            "Posible venta atípica",
            "Producto sano",
            "inventario negativo",
            "producto nuevo sin movimiento"
        ]
        
        winner_field_list = []

        for document in serialized_documents:
            values = {field: document.get(field, float('-inf')) for field in fields_of_interest}            
            valid_values = {key: value for key, value in values.items() if value is not None}            
            if valid_values:
                winner_field = max(valid_values, key=valid_values.get)
            else:                
                winner_field = document["category"] or "Not Valid Value"
            
            winner_field_list.append(winner_field)

        winner_field_counts = dict(Counter(winner_field_list))

        result = {field: winner_field_counts.get(field, 0) for field in fields_of_interest}
        result["No valid field"] = winner_field_counts.get("No Valid Value", 0)

        return {"counts": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/notify")
async def filter_by_fecha(fecha: str = Query(..., description="Filter by 'Fecha' field"),
                           store_id: str = Query(None, description="Filter by 'StoreId' field")):
    try:
        client = MongoClient("mongodb://localhost:27017/") 
        db = client["retail_classificator"]
        collection = db["results"]

        # Construct the query filter
        query = {"Fecha": fecha}
        if store_id:
            query["storeId"] = store_id  # Add storeId filter if provided

        # Fetch the documents with the applied filter
        documents = list(collection.find(query))
        serialized_documents = [serialize_mongo_document(doc) for doc in documents]
        
        if not serialized_documents:
            raise HTTPException(status_code=404, detail="No documents found with the given Fecha and StoreId.")

        fields_of_interest = [
            "Pedido insuficiente",
            "Posible producto eliminando de catalogo",
            "Posible quiebre de stock por pedido insuficiente",
            "Posible venta atípica",
            "Producto sano",
            "inventario negativo",
            "producto nuevo sin movimiento"
        ]
        

        messages = []
        for document in serialized_documents:
            # Extract the values of the fields of interest
            values = {field: document.get(field, float('-inf')) for field in fields_of_interest}            
            valid_values = {key: value for key, value in values.items() if value is not None}            
            if valid_values:
                winner_field = max(valid_values, key=valid_values.get)
            else:
                winner_field = document.get("category", "Not Valid Value")
            
            if winner_field == 'Posible producto eliminando de catalogo':
                messages.append(f"El producto {document['productId']} podria estar eliminandose, por favor revisar su visibilidad para eliminarlo prontamente, ultima venta hace {document['Days Since Last Sale']} dias, stock: {document['Stock dia actual']}, porcentaje de tiendas con stock: {document['% global tiendas']}")
            elif winner_field == 'Posible venta atípica':
                messages.append(f"El producto {document['productId']} podria estar teniendo incremento abruptos en las ventas, por favor revisar la proyeccion y ajustar la cantidad a pedir, ultima venta hace {document['Days Since Last Sale']} dias, stock: {document['Stock dia actual']}, cantidad vendida: {document['Venta dia anterior']}, promedio de ultimas 4 semanas {document['Last 4 Weeks Avg']}")
            elif winner_field == 'Posible quiebre de stock por pedido insuficiente':
                messages.append(f"El producto {document['productId']} podria estar quedandose sin stock suficiente favor de verificar la cantidad pedida, ultima venta hace {document['Days Since Last Sale']} dias, stock: {document['Stock dia actual']}, dias proyectados antes del quiebre de stock {document['Remaining Days']}")
          


        return {"messages":messages}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    
@app.get("/")
def healthCheck():
    return "API online"
        

def preprocess_sample(input_data: pd.DataFrame, features) -> pd.DataFrame:

    X_transformed = preprocessor.transform(input_data)
    num_features = preprocessor.transformers_[0][2]
    all_feature_names = list(num_features)
    X_transformed_df = pd.DataFrame(X_transformed, columns=all_feature_names)
    return X_transformed_df[features]