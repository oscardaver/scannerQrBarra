#!/usr/bin/env python3
"""
Scanner de códigos de barras y QR para Raspberry Pi 3.
Registra fecha, hora, tipo y texto de cada código en un CSV.

Dependencias: python3-opencv, libzbar0, pyzbar
"""
import argparse
import csv
import os
import time
from datetime import datetime

import cv2
from pyzbar import pyzbar

ARCHIVO_CSV = "registro_codigos.csv"
ANCHO, ALTO = 640, 480
SALTAR_FRAMES = 3        # procesa 1 de cada N fotogramas
ESPERA_REPETIDO = 5      # segundos antes de volver a registrar el mismo código


def abrir_csv(ruta):
    nuevo = not os.path.exists(ruta)
    f = open(ruta, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if nuevo:
        w.writerow(["fecha", "hora", "tipo", "texto"])
        f.flush()
    return f, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true",
                    help="no mostrar ventana (ahorra CPU)")
    ap.add_argument("--camara", type=int, default=0, help="índice de la cámara")
    args = ap.parse_args()

    cam = cv2.VideoCapture(args.camara)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cam.isOpened():
        raise SystemExit("No se pudo abrir la cámara")

    archivo, escritor = abrir_csv(ARCHIVO_CSV)
    ultimos = {}   # texto -> momento del último registro
    contador = 0

    print("Escaneando... (Ctrl+C o 'q' para salir)")
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                time.sleep(0.1)
                continue

            contador += 1
            if contador % SALTAR_FRAMES == 0:
                gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                codigos = pyzbar.decode(gris)
                ahora = time.time()

                for c in codigos:
                    texto = c.data.decode("utf-8", errors="replace")
                    tipo = c.type  # QRCODE, EAN13, CODE128, etc.

                    if ahora - ultimos.get(texto, 0) >= ESPERA_REPETIDO:
                        ultimos[texto] = ahora
                        dt = datetime.now()
                        escritor.writerow([
                            dt.strftime("%Y-%m-%d"),
                            dt.strftime("%H:%M:%S"),
                            tipo,
                            texto,
                        ])
                        archivo.flush()
                        print(f"[{dt:%H:%M:%S}] {tipo}: {texto}")

                    if not args.headless:
                        x, y, w, h = c.rect
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, texto, (x, max(y - 8, 15)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if not args.headless:
                cv2.imshow("Scanner", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        archivo.close()
        cv2.destroyAllWindows()
        print("Finalizado.")


if __name__ == "__main__":
    main()
