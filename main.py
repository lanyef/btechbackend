from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import uvicorn
from io import BytesIO
from PIL import Image

app = FastAPI()


loaded_model = tf.keras.models.load_model("./model")

loaded_model.summary()


CLASS_NAMES = ['Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']


@app.get("/")
async def health():
    return {"msg":"server is running, send a post request to /predict with a file to get a response"}

def read_file_as_image(bytes):
    image = Image.open(BytesIO(bytes))
    image = image.resize((256, 256))
    image = image.convert('RGB')
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.post('/predict')
async def prediction(
        file: UploadFile = File(...)
):
    bytes = await file.read()

    image = read_file_as_image(bytes)
    
    predictions = loaded_model.predict(image)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)
