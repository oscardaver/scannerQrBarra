#!/usr/bin/env python3
"""Prueba el intérprete de detectAndDecode y la racha de fotogramas.

No abre la cámara. Si el detector de esta OpenCV no lee el EAN de
pruebas/, el caso se anota como aviso y no tumba la prueba: en 4.6
eso es habitual.

    python3 probar_scanner_opencv.py
"""
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

import scanner_opencv as so

RAIZ = os.path.dirname(os.path.abspath(__file__))


def main():
    fallos = []

    def exigir(condicion, mensaje):
        if condicion:
            print(f"  ok     {mensaje}")
        else:
            fallos.append(mensaje)
            print(f"  FALLO  {mensaje}")

    print("API de 4 valores (OpenCV 4.5–4.7)")
    puntos = np.array([[[10, 20], [110, 22], [108, 80], [12, 78]]], dtype=np.float32)
    lecturas = so.interpretar((
        True,
        ("5901234123457",),
        (int(cv2.barcode.EAN_13),),
        puntos,
    ))
    exigir(len(lecturas) == 1, "un EAN con ok=True")
    if lecturas:
        exigir(lecturas[0].texto == "5901234123457", "el texto es el EAN")
        exigir(lecturas[0].tipo == "EAN13", f"el tipo queda EAN13 ({lecturas[0].tipo})")
        exigir(
            lecturas[0].puntos is not None and lecturas[0].puntos.shape == (4, 2),
            "el polígono tiene 4 vértices",
        )
    exigir(so.interpretar((False, ("5901234123457",), (2,), puntos)) == [],
           "ok=False no se anota")
    exigir(so.interpretar((True, ("",), (2,), None)) == [],
           "texto vacío no se anota")

    print("API de 3 valores (OpenCV 4.8+, sin tipo)")
    lecturas = so.interpretar((True, ("012345678905",), puntos))
    exigir(
        len(lecturas) == 1 and lecturas[0].texto == "012345678905" and lecturas[0].tipo == "BARCODE",
        "3 valores: texto sí, tipo genérico",
    )

    print("API de 3 valores sin el bool")
    lecturas = so.interpretar((("12345670",), (int(cv2.barcode.EAN_8),), puntos.reshape(4, 2)))
    exigir(
        len(lecturas) == 1 and lecturas[0].tipo == "EAN8" and lecturas[0].puntos.shape[0] == 4,
        "3 valores (textos, tipos, puntos) de un solo código",
    )

    print("Racha de fotogramas")
    rachas = so.Rachas(confirmar=3, espera=5)
    lectura = so.Lectura("5901234123457", "EAN13", None)
    exigir(rachas.actualizar([lectura], 0.0) == [], "1 de 3 no anota")
    exigir(rachas.actualizar([lectura], 0.1) == [], "2 de 3 no anota")
    aceptadas = rachas.actualizar([lectura], 0.2)
    exigir(len(aceptadas) == 1 and aceptadas[0].texto == "5901234123457", "3 de 3 anota")
    exigir(rachas.actualizar([lectura], 0.3) == [], "la espera de 5 s frena la repetición")
    rachas.actualizar([], 1.0)
    exigir(rachas.actualizar([lectura], 10.0) == [], "un corte pone la racha a cero")
    rachas.actualizar([lectura], 10.1)
    otra = rachas.actualizar([lectura], 10.2)
    exigir(len(otra) == 1, "otra racha de 3, pasada la espera, vuelve a anotar")

    otra_racha = so.Rachas(2, 5)
    a = so.Lectura("111", "EAN13", None)
    b = so.Lectura("222", "EAN13", None)
    otra_racha.actualizar([a], 0)
    exigir(otra_racha.actualizar([b], 1) == [], "cambiar de código rompe la racha")

    print("Dibujo del polígono")
    lienzo = np.full((120, 160, 3), 255, np.uint8)
    so.dibujar(lienzo, lecturas[0], "12345670")
    verdes = np.count_nonzero(np.all(lienzo == (0, 255, 0), axis=2))
    exigir(verdes > 10, f"el polígono pinta píxeles verdes ({verdes})")

    print("Mensaje si falta el módulo")
    texto_46 = so.mensaje_sin_detector("4.6.0")
    exigir("contrib" in texto_46 and "No lee QR" in texto_46, "4.6 pide el paquete contrib")
    texto_48 = so.mensaje_sin_detector("4.10.0")
    exigir("opencv-python>=4.8" in texto_48, "4.8 pide el paquete principal")

    print("Detector real, sin exigir que acierte")
    try:
        detector = so.crear_detector()
    except SystemExit as exc:
        exigir(False, f"el detector no arranca: {exc}")
        detector = None
    if detector is not None:
        foto = cv2.imread(os.path.join(RAIZ, "pruebas", "ean13.png"))
        exigir(foto is not None, "está la foto de EAN")
        try:
            hallados = so.detectar(detector, foto)
            exigir(isinstance(hallados, list), "detectAndDecode no revienta")
            if hallados:
                print(f"  aviso  esta OpenCV sí leyó: {hallados[0].tipo} {hallados[0].texto}")
            else:
                print(
                    "  aviso  esta OpenCV no leyó el EAN de pruebas/. "
                    "En 4.6 pasa seguido; no es un fallo de la prueba."
                )
        except Exception as exc:
            exigir(False, f"detectar lanzó {exc}")

    print("Línea de comandos")
    ayuda = subprocess.run(
        [sys.executable, os.path.join(RAIZ, "scanner_opencv.py"), "--help"],
        check=False, capture_output=True, text=True,
    )
    exigir(ayuda.returncode == 0 and "--confirmar" in ayuda.stdout, "--help muestra --confirmar")
    with tempfile.TemporaryDirectory() as tmp:
        csv_ruta = os.path.join(tmp, "registro_opencv.csv")
        foto = subprocess.run(
            [sys.executable, os.path.join(RAIZ, "scanner_opencv.py"),
             "--imagen", os.path.join(RAIZ, "pruebas", "ean13.png"),
             "--csv", csv_ruta],
            check=False, capture_output=True, text=True,
        )
        exigir(
            foto.returncode in (0, 1) and "Traceback" not in foto.stderr,
            f"--imagen termina sin traza (código {foto.returncode})",
        )
        exigir("Foto:" in foto.stdout, "la foto avisa que no usa la racha")
        if foto.returncode == 0:
            exigir(
                os.path.exists(csv_ruta) and "5901234123457" in open(csv_ruta, encoding="utf-8").read(),
                "si hubo lectura, el CSV la guarda",
            )

    print()
    if fallos:
        print(f"{len(fallos)} fallo(s):")
        for fallo in fallos:
            print(f"- {fallo}")
        return 1
    print("scanner_opencv.py interpreta la API y confirma la racha.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
