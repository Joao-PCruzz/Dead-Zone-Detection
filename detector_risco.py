import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Point, Polygon

model = YOLO("models/best.pt") # aqui que vai caregar o meu modelo.

# escolhe o vídeo que entra
VIDEO_ENTRADA = "SynthSite/videos/wan2.2-14b_0072.mp4"
VIDEO_SAIDA = "videos/resultado_zona_queda.mp4"

cap = cv2.VideoCapture(VIDEO_ENTRADA)
largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

# Configura gravação do vídeo processado
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(VIDEO_SAIDA, fourcc, fps, (largura, altura))

# Mapeia os nomes das classes direto do seu best.pt
nomes_classes = model.names  # Ex: {0: 'carga', 1: 'person'}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, conf=0.45, verbose=False)

    zonas_risco = []
    trabalhadores = []

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        ids = (
            results[0].boxes.id.cpu().numpy().astype(int)
            if results[0].boxes.id is not None
            else [-1] * len(boxes)
        )

        for box, cls_id, track_id in zip(boxes, classes, ids):
            x1, y1, x2, y2 = map(int, box)
            nome = nomes_classes[cls_id]

            if nome == "carga":
                # Desenha caixa roxa na carga suspensa
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 180), 2)
                cv2.putText(frame, f"CARGA #{track_id}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 180), 2)

                # Projeção geométrica da zona de queda no solo
                cx = (x1 + x2) / 2
                largura_carga = x2 - x1
                # Projeta o centro da carga para o plano do solo abaixo dela
                deslocamento_vertical = int(altura * 0.22)
                cy_solo = min(altura - 10, int(y2 + deslocamento_vertical))

                # Cria uma elipse/polígono de risco proporcional ao tamanho da carga
                raio_x = max(60, int(largura_carga * 0.85))
                raio_y = max(35, int(raio_x * 0.45))  # Achatado pela perspectiva da câmera

                angulos = np.linspace(0, 2 * np.pi, 32)
                pontos_elipse = [
                    (int(cx + raio_x * np.cos(a)), int(cy_solo + raio_y * np.sin(a)))
                    for a in angulos
                ]
                poligono_risco = Polygon(pontos_elipse)
                zonas_risco.append((poligono_risco, pontos_elipse, (int(cx), int(y2)), (int(cx), cy_solo)))

            elif nome == "person":
                # Footpoint: centro da borda inferior da caixa do trabalhador
                foot_x = int((x1 + x2) / 2)
                foot_y = int(y2)
                trabalhadores.append({
                    "id": track_id,
                    "box": (x1, y1, x2, y2),
                    "footpoint": Point(foot_x, foot_y),
                    "foot_xy": (foot_x, foot_y)
                })

    # Desenha as zonas de risco e linhas de projeção vertical
    overlay = frame.copy()
    alerta_ativo = False

    for poly, pts, topo_linha, base_linha in zonas_risco:
        pts_np = np.array(pts, dtype=np.int32)
        cv2.fillPoly(overlay, [pts_np], (0, 165, 255))  # Laranja translúcido
        cv2.polylines(frame, [pts_np], True, (0, 140, 255), 2)
        cv2.line(frame, topo_linha, base_linha, (0, 255, 255), 2, cv2.LINE_AA)

    # Aplica transparência na área de risco no solo
    cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)

    # Verifica interseção geométrica: trabalhador dentro da zona de queda?
    for trab in trabalhadores:
        x1, y1, x2, y2 = trab["box"]
        em_perigo = any(poly.contains(trab["footpoint"]) for poly, _, _, _ in zonas_risco)

        if em_perigo:
            alerta_ativo = True
            cor = (0, 0, 255)  # Vermelho (PERIGO)
            status = f"PERIGO! ID #{trab['id']}"
        else:
            cor = (0, 220, 0)  # Verde (SEGURO)
            status = f"OK #{trab['id']}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
        cv2.circle(frame, trab["foot_xy"], 5, cor, -1)
        cv2.putText(frame, status, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor, 2)

    # Banner superior de alerta geral
    if alerta_ativo:
        cv2.rectangle(frame, (0, 0), (largura, 45), (0, 0, 200), -1)
        cv2.putText(frame, "ALERTA: TRABALHADOR NA ZONA DE QUEDA DE CARGA!", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    out.write(frame)
    cv2.imshow("Monitoramento Zona de Queda", frame)

    # Pressione 'q' para encerrar antes do fim do vídeo
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processamento concluído! Vídeo salvo em: {VIDEO_SAIDA}")