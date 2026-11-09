#!/usr/bin/env python3
"""
Escáner de códigos de barras 1D solo con OpenCV.

Usa cv2.barcode.BarcodeDetector.detectAndDecode. No usa pyzbar ni
preprocesado. No lee QR: el detector nativo cubre EAN y UPC.

El mismo texto tiene que salir en varios fotogramas seguidos antes de
anotarse, para no guardar lecturas sueltas. El registro va a
registro_opencv.csv, aparte del de scanner.py, para poder comparar.

OpenCV 4.8 o más nuevo trae el detector en el paquete principal.
En 4.5, 4.6 y 4.7 está en opencv-contrib (no en opencv-python a secas).
"""
import argparse
import csv
import os
import time
from collections import namedtuple
from datetime import datetime

import cv2
import numpy as np

ARCHIVO_CSV = "registro_opencv.csv"
ANCHO, ALTO = 1280, 720
ESPERA_REPETIDO = 5
CONFIRMAR_DEFECTO = 30

Lectura = namedtuple("Lectura", "texto tipo puntos")

_TIPOS = {
    "NONE": "",
    "EAN_13": "EAN13",
    "EAN13": "EAN13",
    "EAN_8": "EAN8",
    "EAN8": "EAN8",
    "UPC_A": "UPCA",
    "UPCA": "UPCA",
    "UPC_E": "UPCE",
    "UPCE": "UPCE",
    "UPC_EAN_EXTENSION": "EAN-ADDON",
}


def numeros_version(version):
    partes = []
    for bloque in str(version).split("."):
        digitos = ""
        for caracter in bloque:
            if caracter.isdigit():
                digitos += caracter
            else:
                break
        if not digitos:
            break
        partes.append(int(digitos))
    while len(partes) < 3:
        partes.append(0)
    return tuple(partes[:3])


def mensaje_sin_detector(version):
    """Texto de salida cuando falta cv2.barcode.BarcodeDetector."""
    numeros = numeros_version(version)
    if numeros >= (4, 8):
        como = (
            f"Tu OpenCV es {version}. Desde la 4.8 el detector viene en el "
            "paquete principal.\n"
            "En la Raspberry Pi:\n"
            "  sudo apt install python3-opencv\n"
            "Si apt trae una versión anterior, en un entorno virtual:\n"
            '  pip install "opencv-python>=4.8"\n'
        )
    else:
        como = (
            f"Tu OpenCV es {version}. En 4.5, 4.6 y 4.7 el detector no está "
            "en opencv-python: hace falta el paquete contrib.\n"
            "En la Raspberry Pi prueba primero:\n"
            "  sudo apt install python3-opencv\n"
            "(algunas imágenes lo compilan con contrib y otras no).\n"
            "Si sigue sin aparecer, en un entorno virtual:\n"
            '  pip install "opencv-contrib-python>=4.5,<4.8"\n'
            "No instales opencv-python y opencv-contrib-python a la vez; "
            "deja solo el contrib.\n"
            "A partir de OpenCV 4.8 el detector pasó al paquete principal "
            '(pip install "opencv-python>=4.8") y ya no hace falta contrib.\n'
        )
    return (
        "No encuentro cv2.barcode.BarcodeDetector.\n"
        "Este script lee solo códigos de barras 1D (EAN y UPC). No lee QR.\n"
        + como
        + "Para QR y para comparar con pyzbar usa scanner.py, "
        "que no necesita este módulo."
    )


def _clase_detector():
    modulo = getattr(cv2, "barcode", None)
    if modulo is not None and hasattr(modulo, "BarcodeDetector"):
        return modulo.BarcodeDetector
    return getattr(cv2, "barcode_BarcodeDetector", None)


def crear_detector():
    clase = _clase_detector()
    if clase is None:
        raise SystemExit(mensaje_sin_detector(getattr(cv2, "__version__", "?")))
    try:
        return clase()
    except cv2.error as exc:
        raise SystemExit(
            "cv2.barcode.BarcodeDetector está, pero no arrancó "
            f"({exc}).\n" + mensaje_sin_detector(getattr(cv2, "__version__", "?"))
        )


def _como_lista(valor):
    if valor is None:
        return []
    if isinstance(valor, str):
        return [valor]
    if isinstance(valor, np.ndarray):
        if valor.size == 0:
            return []
        if valor.ndim == 0:
            return [valor.item()]
        return [valor[i] for i in range(valor.shape[0])]
    try:
        return list(valor)
    except TypeError:
        return [valor]


def tipo_legible(valor):
    if valor is None:
        return "BARCODE"
    if isinstance(valor, (int, np.integer)):
        modulo = getattr(cv2, "barcode", None)
        if modulo is not None:
            for nombre in (
                "EAN_13", "EAN_8", "UPC_A", "UPC_E", "UPC_EAN_EXTENSION", "NONE",
            ):
                constante = getattr(modulo, nombre, None)
                if constante is not None and int(constante) == int(valor):
                    valor = nombre
                    break
            else:
                return "BARCODE"
        else:
            return "BARCODE"
    texto = str(valor).strip()
    if not texto or texto.upper() == "NONE":
        return "BARCODE"
    return _TIPOS.get(texto.upper(), texto.upper().replace("-", "").replace(" ", ""))


def puntos_de(puntos, indice, cuantos):
    if puntos is None:
        return None
    arreglo = np.asarray(puntos, dtype=np.float32)
    if arreglo.size == 0:
        return None
    if arreglo.ndim == 2 and arreglo.shape[-1] == 2:
        if cuantos <= 1:
            return arreglo.reshape(-1, 2)
        if arreglo.shape[0] == cuantos * 4:
            return arreglo.reshape(cuantos, 4, 2)[indice]
        return arreglo
    if arreglo.ndim >= 3 and arreglo.shape[-1] == 2 and indice < arreglo.shape[0]:
        return arreglo[indice].reshape(-1, 2)
    return None


def interpretar(resultado):
    """Normaliza lo que devuelve detectAndDecode según la versión.

    OpenCV 4.5–4.7 (contrib) suele dar 4 valores:
    (ok, textos, tipos, puntos).
    OpenCV 4.8 o más nuevo suele dar 3:
    (ok, textos, puntos), sin el tipo.
    Algunas compilaciones omiten el bool y dan (textos, tipos, puntos).
    """
    if resultado is None:
        return []
    if not isinstance(resultado, tuple):
        resultado = (resultado,)

    ok = True
    textos = []
    tipos = []
    puntos = None

    if len(resultado) == 4:
        ok, textos, tipos, puntos = resultado
    elif len(resultado) == 3:
        primero, segundo, tercero = resultado
        if isinstance(primero, (bool, np.bool_)):
            ok, textos, puntos = primero, segundo, tercero
        else:
            textos, tipos, puntos = primero, segundo, tercero
    elif len(resultado) == 2:
        primero, segundo = resultado
        if isinstance(primero, (bool, np.bool_)):
            ok, textos = primero, segundo
        else:
            textos, puntos = primero, segundo
    elif len(resultado) == 1:
        textos = resultado[0]
    else:
        return []

    if ok is False or ok is np.False_:
        return []

    lista_textos = _como_lista(textos)
    lista_tipos = _como_lista(tipos)
    cuantos = len(lista_textos)
    lecturas = []
    vistos = set()
    for indice, texto in enumerate(lista_textos):
        if isinstance(texto, bytes):
            texto = texto.decode("utf-8", errors="replace")
        texto = str(texto).strip()
        if not texto:
            continue
        tipo = tipo_legible(lista_tipos[indice] if indice < len(lista_tipos) else None)
        clave = (tipo, texto)
        if clave in vistos:
            continue
        vistos.add(clave)
        lecturas.append(Lectura(texto, tipo, puntos_de(puntos, indice, cuantos)))
    return lecturas


def detectar(detector, frame):
    """Una pasada del detector, sin preprocesar el fotograma."""
    metodos = []
    if hasattr(detector, "detectAndDecodeWithType"):
        metodos.append(detector.detectAndDecodeWithType)
    metodos.append(detector.detectAndDecode)
    ultimo_error = None
    for metodo in metodos:
        try:
            return interpretar(metodo(frame))
        except cv2.error as exc:
            ultimo_error = exc
    if ultimo_error is not None:
        print(f"OpenCV no pudo leer el fotograma: {ultimo_error}")
    return []


class Rachas:
    """Exige el mismo código en N fotogramas seguidos y luego espera."""

    def __init__(self, confirmar, espera):
        self.confirmar = max(1, int(confirmar))
        self.espera = float(espera)
        self.cuenta = {}
        self.ultimo = {}

    def actualizar(self, lecturas, ahora):
        presentes = {}
        for lectura in lecturas:
            presentes[(lectura.tipo, lectura.texto)] = lectura
        for clave in list(self.cuenta):
            if clave not in presentes:
                self.cuenta[clave] = 0
        aceptadas = []
        for clave, lectura in presentes.items():
            self.cuenta[clave] = self.cuenta.get(clave, 0) + 1
            if self.cuenta[clave] < self.confirmar:
                continue
            previo = self.ultimo.get(clave)
            if previo is not None and ahora - previo < self.espera:
                continue
            self.ultimo[clave] = ahora
            self.cuenta[clave] = 0
            aceptadas.append(lectura)
        return aceptadas

    def avance(self, lectura):
        return self.cuenta.get((lectura.tipo, lectura.texto), 0)


def abrir_csv(ruta):
    nuevo = not os.path.exists(ruta)
    archivo = open(ruta, "a", newline="", encoding="utf-8")
    escritor = csv.writer(archivo)
    if nuevo:
        escritor.writerow(["fecha", "hora", "tipo", "texto"])
        archivo.flush()
    return archivo, escritor


def anotar(lecturas, escritor, archivo):
    momento = datetime.now()
    for lectura in lecturas:
        escritor.writerow([
            momento.strftime("%Y-%m-%d"),
            momento.strftime("%H:%M:%S"),
            lectura.tipo,
            lectura.texto,
        ])
        archivo.flush()
        print(f"[{momento:%H:%M:%S}] {lectura.tipo}: {lectura.texto}")


def dibujar(frame, lectura, etiqueta):
    alto, ancho = frame.shape[:2]
    origen = (20, 30)
    if lectura.puntos is not None:
        pts = np.asarray(lectura.puntos, dtype=np.int32).reshape(-1, 2)
        if len(pts) >= 2:
            cv2.polylines(frame, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
            origen = (int(pts[0, 0]), int(pts[0, 1]))
    x = max(0, min(ancho - 1, origen[0]))
    y = max(15, min(alto - 1, origen[1] - 8))
    cv2.putText(
        frame,
        etiqueta,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )


def mostrar(frame):
    alto, ancho = frame.shape[:2]
    vista = frame
    if ancho > 800:
        nuevo_w = 800
        nuevo_h = max(1, int(round(alto * nuevo_w / ancho)))
        vista = cv2.resize(frame, (nuevo_w, nuevo_h), interpolation=cv2.INTER_AREA)
    cv2.imshow("Barras OpenCV", vista)
    return (cv2.waitKey(1) & 0xFF) == ord("q")


def configurar_camara(cam, ancho, alto, mjpg):
    if mjpg:
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
    try:
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except cv2.error:
        pass


def _abrir_dispositivo(indice):
    if hasattr(cv2, "CAP_V4L2"):
        cam = cv2.VideoCapture(indice, cv2.CAP_V4L2)
        if cam.isOpened():
            return cam
        cam.release()
    cam = cv2.VideoCapture(indice)
    if not cam.isOpened():
        cam.release()
        raise SystemExit("No se pudo abrir la cámara")
    return cam


def _hay_imagen(cam):
    for _ in range(10):
        ok, frame = cam.read()
        if ok and frame is not None and frame.size:
            return frame
        time.sleep(0.05)
    return None


def abrir_camara(indice, ancho, alto, mjpg):
    cam = _abrir_dispositivo(indice)
    configurar_camara(cam, ancho, alto, mjpg)
    frame = _hay_imagen(cam)
    if frame is None and (ancho, alto, mjpg) != (640, 480, False):
        print(
            f"La cámara no entregó imagen a {ancho}x{alto}"
            f"{' MJPG' if mjpg else ''}. Reintento a 640x480."
        )
        cam.release()
        return abrir_camara(indice, 640, 480, False)
    if frame is None:
        cam.release()
        raise SystemExit("La cámara no entregó imágenes")
    return cam, frame.shape[1], frame.shape[0]


def procesar_foto(ruta, detector, csv_ruta):
    frame = cv2.imread(ruta, cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"No se pudo leer la imagen: {ruta}")
    print(
        "Foto: se anota lo que salga en este único fotograma. "
        "La confirmación de varios fotogramas vale para la cámara."
    )
    lecturas = detectar(detector, frame)
    if not lecturas:
        print(
            "No se leyó ningún código.\n"
            "Este detector solo cubre EAN y UPC, no QR ni Code 128. "
            "En OpenCV 4.6 a veces tampoco lee un EAN nítido. "
            "Compara con: python3 scanner.py --imagen " + ruta
        )
        return 1
    archivo, escritor = abrir_csv(csv_ruta)
    try:
        anotar(lecturas, escritor, archivo)
    finally:
        archivo.close()
    return 0


def bucle(args, detector):
    cam, ancho_real, alto_real = abrir_camara(
        args.camara, args.ancho, args.alto, args.mjpg
    )
    if ancho_real < args.ancho * 0.9 or alto_real < args.alto * 0.9:
        print(
            f"Aviso: la cámara entregó {ancho_real}x{alto_real} "
            f"(se pidió {args.ancho}x{args.alto})."
        )
    else:
        print(f"Cámara {ancho_real}x{alto_real}.")
    print(
        f"OpenCV {cv2.__version__}. Solo EAN/UPC, no QR. "
        f"Confirma {args.confirmar} fotogramas seguidos. "
        f"Espera {args.espera:g} s. CSV: {args.csv}. "
        "Ctrl+C o 'q' para salir."
    )

    archivo, escritor = abrir_csv(args.csv)
    rachas = Rachas(args.confirmar, args.espera)
    headless = args.headless
    inicio = time.time()
    capturas = 0
    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            capturas += 1
            ahora = time.time()
            lecturas = detectar(detector, frame)
            aceptadas = rachas.actualizar(lecturas, ahora)
            if aceptadas:
                anotar(aceptadas, escritor, archivo)
            if not headless:
                for lectura in lecturas:
                    avance = rachas.avance(lectura)
                    if avance <= 0:
                        etiqueta = lectura.texto
                    else:
                        visto = min(avance, args.confirmar)
                        etiqueta = f"{lectura.texto} ({visto}/{args.confirmar})"
                    dibujar(frame, lectura, etiqueta)
                try:
                    if mostrar(frame):
                        break
                except cv2.error:
                    print("No hay ventana; sigo sin imagen.")
                    headless = True
            if time.time() - inicio >= 5:
                elapsed = time.time() - inicio
                print(
                    f"ritmo {capturas / elapsed:.1f} capturas/s "
                    f"({ancho_real}x{alto_real})"
                )
                inicio = time.time()
                capturas = 0
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        archivo.close()
        cv2.destroyAllWindows()
        print("Finalizado.")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Lee códigos de barras 1D (EAN/UPC) con el detector nativo "
            "de OpenCV y los anota en un CSV. No lee QR."
        )
    )
    ap.add_argument("--headless", action="store_true",
                    help="no mostrar ventana (ahorra CPU en la Pi)")
    ap.add_argument("--camara", type=int, default=0, help="índice de la cámara")
    ap.add_argument("--ancho", type=int, default=ANCHO,
                    help=f"ancho pedido a la cámara (default {ANCHO})")
    ap.add_argument("--alto", type=int, default=ALTO,
                    help=f"alto pedido a la cámara (default {ALTO})")
    ap.add_argument("--mjpg", action="store_true",
                    help="pide MJPEG a la cámara (menos USB; a veces más compresión)")
    ap.add_argument("--csv", default=ARCHIVO_CSV,
                    help=f"registro, aparte del de scanner.py (default {ARCHIVO_CSV})")
    ap.add_argument("--espera", type=float, default=ESPERA_REPETIDO,
                    help="segundos antes de volver a anotar el mismo texto")
    ap.add_argument("--confirmar", type=int, default=CONFIRMAR_DEFECTO,
                    help="fotogramas seguidos con el mismo texto antes de anotar "
                         f"(default {CONFIRMAR_DEFECTO})")
    ap.add_argument("--imagen",
                    help="decodifica una foto y termina, sin abrir la cámara")
    args = ap.parse_args()
    if args.confirmar < 1:
        ap.error("--confirmar debe ser >= 1")
    if args.espera < 0:
        ap.error("--espera debe ser >= 0")
    detector = crear_detector()
    if args.imagen:
        raise SystemExit(procesar_foto(args.imagen, detector, args.csv))
    bucle(args, detector)


if __name__ == "__main__":
    main()
