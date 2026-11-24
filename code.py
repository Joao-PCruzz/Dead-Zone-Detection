from ultralytics import YOLO

# Baixa automaticamente o modelo Nano (~6 MB) e testa no seu vídeo
model = YOLO("yolo11n.pt")
model.predict(source="SynthSite/videos/sora-2-pro_0001.mp4", show=True, classes=[0]) # classes=[0] filtra só pessoas