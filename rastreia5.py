import cv2
from ultralytics import YOLO
from math import floor
import time
import csv
import numpy as np

#otimização de quadros
skip_frames = 2
frame_count = 0

# tempo de sinal aberto
anterior = 0
intervalos_abertos = []
n_temp = 0

#carrega YOLOv8 
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("video/video2.mp4")

#width e height do video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def sinal(frame, width, height):
    y1 = floor(height*0.17)
    y2 = floor(height*0.23)
    x1 = floor(width*0.59)
    x2 = floor(width*0.61)

    #area do semaforo
    semaforo_roi = frame[y1:y2,x1:x2] 

    # Converter para HSV
    hsv = cv2.cvtColor(semaforo_roi, cv2.COLOR_BGR2HSV)

    # Máscaras para vermelho (dois ranges no HSV)
    lower_red1 = (0, 100, 100)
    upper_red1 = (10, 255, 255)
    lower_red2 = (160, 100, 100)
    upper_red2 = (179, 255, 255)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Checa se tem vermelho suficiente
    red_pixels = cv2.countNonZero(red_mask)
    total_pixels = (y2-y1)*(x2-x1)
    print(total_pixels)
    red_ratio = red_pixels / total_pixels

    semaforo_vermelho = red_ratio > 0.1  

    # Mostra o status do semáforo no frame
    status = "Semaforo: VERMELHO" if semaforo_vermelho else "Semaforo: NAO vermelho"

    cor_status = (0, 0, 255) if semaforo_vermelho else (0, 255, 0)
    cv2.putText(frame, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor_status, 2)

    #desenha a região do semáforo
    cv2.rectangle(frame, (floor(width*0.59), floor(height*0.17)), (floor(width*0.61), floor(height*0.23)), cor_status, 2)

    # a cor do semaforo
    return semaforo_vermelho


# Define as zonas de interesse (x_min, y_min, x_max, y_max)
zone = ((floor(width*0.32)), (floor(height*0.65)), (floor(width*0.7)), (floor(height*0.66)))  # Exemplo: retângulo na parte inferior

zone2 = ((floor(width*0.55)), (floor(height*0.55)), (floor(width*0.90)), (floor(height*0.56)))  # Exemplo: retângulo na parte inferior

# Set para armazenar IDs de carros que já entraram na zona e já sairam
carros_na_zona = set()
quantidade = 0

color = (0,0,0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % skip_frames != 0:
        continue  # pula este frame
    
    # coordenadas do polígono  sup-esq (100, 50), sup-dir (200, 80), inf-dir (150, 200), inf-esq (150, 200)
    pts = np.array([
    [floor(width*0.1), floor(height*0.1)], # ponto superior esquerdo 
    [floor(width*0.5), 0], # ponto superior direito
    [floor(width*0.6), floor(height*0.35)], # ponto inferior direito
    [-floor(width*0.2), height], # ponto inferior direito
    [-floor(width*0.4), height] # ponto inferior esquerdo
], dtype=np.int32) 
    pts = pts.reshape((-1, 1, 2))

    # Fill the polygon
    cv2.fillPoly(frame, [pts], color)

    semaforo_vermelho = sinal(frame, width, height)  # Chama a função para verificar o semáforo

    results = model.track(frame, persist=True)
    boxes = results[0].boxes

    # Desenhar a zonas
    cv2.rectangle(frame, (zone[0], zone[1]), (zone[2], zone[3]), (255, 0, 0), 2) # azul
    cv2.rectangle(frame, (zone2[0], zone2[1]), (zone2[2], zone2[3]), (0, 255, 0), 2) # verde

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id == 2 or cls_id == 7:  # classe 'car' 
                x1, y1, x2, y2 = map(int, box.xyxy[0]) # coordenadas da bounding box
                conf = float(box.conf[0])

                # ID do objeto (para tracking)
                track_id = int(box.id[0]) if box.id is not None else None
                if track_id is None:
                    continue

                # Se entrou na zona, salva o ID
                if (y2 >= zone[1]) and (y1 <= zone[3]) and ((x1+x2)//2 >= zone[0]) and ((x1+x2)//2 <= zone[2]) and conf >= 0.7:
                    carros_na_zona.add(track_id)

                # Se o ID estiver registrado, desenha
                if track_id in carros_na_zona:
                    label = f"Car {track_id}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Carro atravessou o semaforo
                if (y2 <= zone2[3]  and ((x1+x2)//2 >= zone2[0]) and (track_id in carros_na_zona) and not semaforo_vermelho):
                    carros_na_zona.remove(track_id)
                    quantidade += 1
                    print("Carro passou na zona 2")

                if semaforo_vermelho and anterior == 1:
                    print("Semáforo vermelho detectado")
                    # se o semaforo ficou vermelho para o tempo
                    to = time.time() - n_temp
                    # Adiciona o intervalo e a quantidade de carros ao CSV
                    intervalos_abertos.append((to, quantidade, (quantidade/to)*60))
                    #estado anterior passa a ser vermelho
                    anterior = 0
                elif not semaforo_vermelho and anterior == 0:
                    print("Semáforo não vermelho detectado")
                    # Se o semáforo não estiver vermelho, anterior passa a ser verde e inicia contagem de tempo
                    anterior = 1
                    n_temp = time.time()

    cv2.imshow("Filtered and Zoned", frame)

    # Pressione 'q' para sair do loop (fechar a janela)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        if not semaforo_vermelho: #caso o video acabe e o semaforo esteja não-vermelho
            # se o semaforo ficou vermelho para o tempo
            to = time.time() - n_temp
            # Adiciona o intervalo e a quantidade de carros ao CSV
            intervalos_abertos.append((to, quantidade, (quantidade/to)*60))
        break


cap.release()
cv2.destroyAllWindows()

if not semaforo_vermelho: #caso o video acabe e o semaforo esteja não-vermelho
            # se o semaforo ficou vermelho para o tempo
            to = time.time() - n_temp
            # Adiciona o intervalo e a quantidade de carros ao CSV
            intervalos_abertos.append((to, quantidade, (quantidade/to)*60))

# Salvar CSV ao final
with open("sinal_aberto.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Tempo_sinal_aberto(s)", "quantidade_carros", "carros_minuto"])
    for temp, quant, cpm in intervalos_abertos:
        writer.writerow([temp, quant, cpm])

print("CSV salvo como 'sinal_aberto.csv'")