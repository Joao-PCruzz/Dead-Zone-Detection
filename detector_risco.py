import os
from collections import defaultdict, deque
import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon, LineString

# 1. Carrega o modelo treinado
model = YOLO("models/best.pt")

# Caminhos de entrada e saída
VIDEO_ENTRADA = "SynthSite/videos/veo3.1_0099.mp4"
VIDEO_SAIDA = "videos/resultado_zona_queda.mp4"
os.makedirs(os.path.dirname(VIDEO_SAIDA), exist_ok=True)

# Limiares de confiança específicos por classe
CONF_CARGA = 0.22
CONF_PERSON = 0.45

# Parâmetros físicos e de movimento
ALTURA_MEDIA_TRABALHADOR_M = 1.75
ALTURA_SEGURA_MANUSEIO_M = 0.50  # <= 0.5m do solo: carga segura para manuseio/guia
LIMIAR_MOVIMENTO_PX = 2.5        # Deslocamento mínimo (px) em 4 frames para considerar em movimento

cap = cv2.VideoCapture(VIDEO_ENTRADA)
largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

# Configura gravação do vídeo processado
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(VIDEO_SAIDA, fourcc, fps, (largura, altura))

# Mapeia os nomes das classes direto do seu best.pt
nomes_classes = model.names  # Ex: {0: 'carga', 1: 'person'}

# Histórico de centros por track_id da carga para detectar movimento
historico_cargas = defaultdict(lambda: deque(maxlen=10))


def verificar_movimento_carga(track_id, cx, cy):
    """Retorna (em_movimento, dx, dy) comparando a posição atual com 4 frames atrás."""
    if track_id == -1:
        return False, 0.0, 0.0

    hist = historico_cargas[track_id]
    hist.append((cx, cy))

    if len(hist) < 4:
        return False, 0.0, 0.0

    ant_x, ant_y = hist[-4]
    dx = cx - ant_x
    dy = cy - ant_y
    deslocamento = np.hypot(dx, dy)

    return deslocamento > LIMIAR_MOVIMENTO_PX, dx, dy


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Roda inferência com ByteTrack usando o menor limiar para não perder cargas difíceis
    results = model.track(frame, persist=True, conf=min(CONF_CARGA, CONF_PERSON), verbose=False)

    cargas_detectadas = []
    trabalhadores = []

    # 1ª Passagem: separa trabalhadores e cargas detectadas no frame
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()
        ids = (
            results[0].boxes.id.cpu().numpy().astype(int)
            if results[0].boxes.id is not None
            else [-1] * len(boxes)
        )

        for box, cls_id, track_id, conf in zip(boxes, classes, ids, confs):
            x1, y1, x2, y2 = map(int, box)
            nome = nomes_classes[cls_id]

            if nome == "carga" and conf >= CONF_CARGA:
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                em_movimento, dx, dy = verificar_movimento_carga(track_id, cx, cy)

                cargas_detectadas.append({
                    "id": track_id,
                    "box": (x1, y1, x2, y2),
                    "conf": conf,
                    "cx": cx,
                    "cy": cy,
                    "em_movimento": em_movimento,
                    "dx": dx,
                    "dy": dy
                })

            elif nome == "person" and conf >= CONF_PERSON:
                alt_pessoa = y2 - y1
                larg_pessoa = x2 - x1

                # Sobe 10% da altura da caixa (região dos tornozelos/base real no solo)
                y_contato = int(y2 - 0.10 * alt_pessoa)

                # Segmento horizontal representando a largura da pisada (60% da largura da caixa)
                margem_x = int(0.20 * larg_pessoa)
                pt_esq = (x1 + margem_x, y_contato)
                pt_dir = (x2 - margem_x, y_contato)

                trabalhadores.append({
                    "id": track_id,
                    "box": (x1, y1, x2, y2),
                    "cx": (x1 + x2) / 2.0,
                    "altura": alt_pessoa,
                    "y_solo": y_contato,
                    "linha_pes": LineString([pt_esq, pt_dir]),
                    "pt_esq": pt_esq,
                    "pt_dir": pt_dir
                })

    # 2ª Passagem: estima altura da carga usando trabalhadores como régua e projeta zona de queda
    zonas_risco = []
    zonas_manuseio = []

    for carga in cargas_detectadas:
        x1, y1, x2, y2 = carga["box"]
        track_id = carga["id"]
        conf = carga["conf"]
        cx = carga["cx"]
        cy = carga["cy"]
        em_movimento = carga["em_movimento"]
        largura_carga = x2 - x1

        if trabalhadores:
            # Prioriza trabalhadores horizontalmente próximos da carga para evitar erro de perspectiva
            proximos = [
                t for t in trabalhadores
                if abs(t["cx"] - cx) <= max(largura * 0.30, largura_carga * 1.5)
            ]
            ref_grupo = proximos if proximos else trabalhadores

            alt_ref_px = float(np.median([t["altura"] for t in ref_grupo]))
            y_solo_ref = float(np.median([t["y_solo"] for t in ref_grupo]))

            # Distância vertical em pixels da base da carga até o nível do solo estimado
            dist_solo_px = max(0.0, y_solo_ref - y2)
            altura_m = (dist_solo_px / max(alt_ref_px, 1.0)) * ALTURA_MEDIA_TRABALHADOR_M

            # Projeta o centro da zona no plano do solo estimado
            cy_solo = int(np.clip(y_solo_ref, y2 + 5, altura - 10))
        else:
            # Fallback quando não há trabalhadores no frame
            dist_solo_px = altura * 0.12
            altura_m = ALTURA_MEDIA_TRABALHADOR_M
            cy_solo = min(altura - 10, int(y2 + dist_solo_px))

        # Verifica se a carga está dentro da altura segura para manuseio (<= 0.5m)
        segura_manuseio = altura_m <= ALTURA_SEGURA_MANUSEIO_M

        if segura_manuseio:
            nivel_alt = "MANUSEIO"
            fator_pendulo = 1.0
            cor_normal = (255, 200, 0)  # Ciano para altura segura de guia
        elif altura_m < 1.5:
            nivel_alt = "BAIXA"
            fator_pendulo = 1.0
            cor_normal = (255, 0, 180)  # Roxo padrão
        elif altura_m < 4.0:
            nivel_alt = "MEDIA"
            fator_pendulo = 1.15
            cor_normal = (255, 0, 180)
        else:
            nivel_alt = "ALTA"
            fator_pendulo = 1.30
            cor_normal = (255, 0, 180)

        # Define cor da borda e identificador de movimento
        if em_movimento:
            cor_caixa_carga = (0, 0, 255)  # Vermelho quando em movimento
            status_mov = "MOVENDO"
        else:
            cor_caixa_carga = cor_normal     # Cor normal quando parada
            status_mov = "PARADA"

        # Desenha caixa da carga + seta direcional (se movendo) + rótulo completo
        cv2.rectangle(frame, (x1, y1), (x2, y2), cor_caixa_carga, 2)
        if em_movimento:
            pt_ini = (int(cx), int(cy))
            pt_fim = (int(cx + carga["dx"] * 4.0), int(cy + carga["dy"] * 4.0))
            cv2.arrowedLine(frame, pt_ini, pt_fim, (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.35)

        label_carga = f"CARGA #{track_id} ({conf:.2f}) | ~{altura_m:.1f}m [{nivel_alt}] [{status_mov}]"
        cv2.putText(frame, label_carga, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, cor_caixa_carga, 2)

        # Cria elipse/polígono proporcional ao tamanho da carga e à altura (pêndulo)
        raio_x = int(max(65, largura_carga * 0.75) * fator_pendulo)
        raio_y = max(45, int(raio_x * 0.55))

        angulos = np.linspace(0, 2 * np.pi, 32)
        pontos_elipse = [
            (int(cx + raio_x * np.cos(a)), int(cy_solo + raio_y * np.sin(a)))
            for a in angulos
        ]
        poligono = Polygon(pontos_elipse)

        # Separa zonas de manuseio seguro das zonas ativas de risco de queda
        if segura_manuseio:
            zonas_manuseio.append((poligono, pontos_elipse, (int(cx), int(y2)), (int(cx), cy_solo)))
        else:
            zonas_risco.append((poligono, pontos_elipse, (int(cx), int(y2)), (int(cx), cy_solo)))

    # Desenha as zonas no solo (ciano para manuseio seguro, laranja para risco de queda)
    overlay = frame.copy()
    alerta_ativo = False

    for poly, pts, topo_linha, base_linha in zonas_manuseio:
        pts_np = np.array(pts, dtype=np.int32)
        cv2.fillPoly(overlay, [pts_np], (255, 200, 0))
        cv2.polylines(frame, [pts_np], True, (255, 180, 0), 2)

    for poly, pts, topo_linha, base_linha in zonas_risco:
        pts_np = np.array(pts, dtype=np.int32)
        cv2.fillPoly(overlay, [pts_np], (0, 165, 255))
        cv2.polylines(frame, [pts_np], True, (0, 140, 255), 2)
        cv2.line(frame, topo_linha, base_linha, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)

    # Verifica interseção geométrica APENAS contra zonas de risco ativas (> 0.5m)
    for trab in trabalhadores:
        x1, y1, x2, y2 = trab["box"]
        em_perigo = any(poly.intersects(trab["linha_pes"]) for poly, _, _, _ in zonas_risco)

        if em_perigo:
            alerta_ativo = True
            cor = (0, 0, 255)  # Vermelho (PERIGO)
            status = f"PERIGO! ID #{trab['id']}"
        else:
            cor = (0, 220, 0)  # Verde (SEGURO)
            status = f"OK #{trab['id']}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
        cv2.line(frame, trab["pt_esq"], trab["pt_dir"], cor, 3)
        cv2.putText(frame, status, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor, 2)

    # Banner superior de alerta geral
    if alerta_ativo:
        cv2.rectangle(frame, (0, 0), (largura, 45), (0, 0, 200), -1)
        cv2.putText(frame, "ALERTA: TRABALHADOR NA ZONA DE QUEDA DE CARGA!", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    out.write(frame)
    cv2.imshow("Monitoramento Zona de Queda", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processamento concluído! Vídeo salvo em: {VIDEO_SAIDA}")