# server.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import uvicorn
import os
import uuid

app = FastAPI()

model = YOLO("yolo11n.pt")

OUTPUT_DIR = "/ultralytics/runs/web_predict"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 這個要改成「瀏覽器真的連得到」的 Base URL
PUBLIC_BASE_URL = "http://192.168.50.82:6611"  # 若 WebUI 不是在同一台，就改成對你瀏覽器可見的 host/ip

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        img = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot read image file")

    # 1) 不用 save=True，自行控制輸出
    results = model.predict(img, save=False)
    det = results[0]

    # 2) 自己決定輸出檔名
    run_id = str(uuid.uuid4())
    filename = f"{run_id}.jpg"
    save_path = os.path.join(OUTPUT_DIR, filename)

    # Ultralytics 的 Results.save 會幫你把標註畫到圖上並存檔:contentReference[oaicite:2]{index=2}
    det.save(save_path)

    # 3) 組成可被瀏覽器存取的 URL
    annotated_url = f"{PUBLIC_BASE_URL}/annotated/{filename}"

    out = []
    for box in det.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = [float(x) for x in box.xyxy[0].tolist()]
        out.append({
            "class_id": cls,
            "class_name": det.names[cls],
            "confidence": conf,
            "bbox_xyxy": xyxy
        })

    return JSONResponse({
        "detections": out,
        "annotated_image_url": annotated_url
    })


@app.get("/annotated/{filename}")
async def get_annotated(filename: str):
    # 讓瀏覽器可以直接拿到結果圖片
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6611)
