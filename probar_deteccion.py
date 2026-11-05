#!/usr/bin/env python3
"""Compara la lectura antigua (gris + pyzbar una vez) con el escáner nuevo.

No usa la cámara. Genera escenas a partir de las fotos en pruebas/ y
comprueba dos cosas: que los QR que ya se leían sigan saliendo por la
vía rápida, y que varios códigos 1D que antes se perdían ahora se lean.

    python3 probar_deteccion.py
"""
import csv
import io
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from pyzbar.pyzbar import decode

import scanner

RAIZ = os.path.dirname(os.path.abspath(__file__))
ESPERADO = {
    "ean13": "5901234123457",
    "code128": "ABC-12345",
    "qr": "https://ejemplo.test/p/123",
}
TIPO = {"ean13": "EAN13", "code128": "CODE128", "qr": "QRCODE"}


def cargar(nombre):
    ruta = os.path.join(RAIZ, "pruebas", f"{nombre}.png")
    imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        raise SystemExit(f"Falta la fixtura {ruta}")
    return imagen


def linea_base(gris):
    textos = []
    for codigo in decode(gris):
        textos.append(codigo.data.decode("utf-8", errors="replace").strip())
    return textos


def colocar(simbolo, ancho_imagen, alto_imagen, ancho_codigo, angulo=0,
            blur=0.0, contraste=1.0, brillo=0.0, x_frac=0.5, y_frac=0.5):
    """Pega el símbolo ya en la resolución final (desenfoque después)."""
    lienzo = np.full((alto_imagen, ancho_imagen), 236, np.uint8)
    relacion = ancho_codigo / float(simbolo.shape[1])
    tw = max(8, int(round(ancho_codigo)))
    th = max(8, int(round(simbolo.shape[0] * relacion)))
    barra = cv2.resize(
        simbolo,
        (tw, th),
        interpolation=cv2.INTER_AREA if relacion < 1 else cv2.INTER_LINEAR,
    )
    if abs(angulo) > 0.1:
        matriz = cv2.getRotationMatrix2D((tw / 2.0, th / 2.0), angulo, 1.0)
        coseno, seno = abs(matriz[0, 0]), abs(matriz[0, 1])
        rw = int(th * seno + tw * coseno)
        rh = int(th * coseno + tw * seno)
        matriz[0, 2] += rw / 2.0 - tw / 2.0
        matriz[1, 2] += rh / 2.0 - th / 2.0
        barra = cv2.warpAffine(
            barra, matriz, (rw, rh), flags=cv2.INTER_LINEAR, borderValue=236
        )
    bh, bw = barra.shape[:2]
    x = int(x_frac * ancho_imagen - bw / 2)
    y = int(y_frac * alto_imagen - bh / 2)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(ancho_imagen, x + bw), min(alto_imagen, y + bh)
    if x1 > x0 and y1 > y0:
        lienzo[y0:y1, x0:x1] = barra[y0 - y:y1 - y, x0 - x:x1 - x]
    if contraste != 1.0 or brillo:
        lienzo = np.clip(
            lienzo.astype(np.float32) * contraste + brillo, 0, 255
        ).astype(np.uint8)
    if blur > 0:
        lienzo = cv2.GaussianBlur(lienzo, (0, 0), blur)
    return lienzo


def escena_optica(simbolo, ancho_px_en_640, salida, angulo=0, blur=0.0):
    """Misma fracción del encuadre a otra resolución, con desenfoque óptico.

    El desenfoque se aplica antes de muestrear, como en la cámara. Ampliar
    después no recupera barras que el sensor ya fusionó; más píxeles sí.
    """
    out_w, out_h = salida
    factor = ancho_px_en_640 / 640.0
    escala = 3
    ancho, alto = out_w * escala, out_h * escala
    lienzo = np.full((alto, ancho), 236, np.uint8)
    tw = max(8, int(round(factor * ancho)))
    th = max(8, int(round(simbolo.shape[0] * (tw / simbolo.shape[1]))))
    barra = cv2.resize(simbolo, (tw, th), interpolation=cv2.INTER_AREA)
    if abs(angulo) > 0.1:
        matriz = cv2.getRotationMatrix2D((tw / 2.0, th / 2.0), angulo, 1.0)
        coseno, seno = abs(matriz[0, 0]), abs(matriz[0, 1])
        rw = int(th * seno + tw * coseno)
        rh = int(th * coseno + tw * seno)
        matriz[0, 2] += rw / 2.0 - tw / 2.0
        matriz[1, 2] += rh / 2.0 - th / 2.0
        barra = cv2.warpAffine(
            barra, matriz, (rw, rh), flags=cv2.INTER_LINEAR, borderValue=236
        )
    bh, bw = barra.shape[:2]
    x = ancho // 2 - bw // 2
    y = alto // 2 - bh // 2
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(ancho, x + bw), min(alto, y + bh)
    lienzo[y0:y1, x0:x1] = barra[y0 - y:y1 - y, x0 - x:x1 - x]
    if blur > 0:
        lienzo = cv2.GaussianBlur(lienzo, (0, 0), blur * escala)
    return cv2.resize(lienzo, (out_w, out_h), interpolation=cv2.INTER_AREA)


def evaluar(gris, simbolos):
    base = set(linea_base(gris))
    lecturas = scanner.leer_hasta_encontrar(gris, simbolos)
    return base, lecturas


def main():
    simbolos = scanner.simbolos_zbar(scanner.SIMBOLOS_DEFECTO)
    fuentes = {nombre: cargar(nombre) for nombre in ESPERADO}
    fallos = []

    def exigir(condicion, mensaje):
        if not condicion:
            fallos.append(mensaje)
            print(f"  FALLO  {mensaje}")
        else:
            print(f"  ok     {mensaje}")

    print("Fixturas nítidas")
    for nombre, imagen in fuentes.items():
        textos = set(linea_base(imagen))
        exigir(
            ESPERADO[nombre] in textos,
            f"{nombre} nítido se lee ({sorted(textos)[:2]})",
        )

    print("Confirmación antes de anotar")
    buffer = io.StringIO()
    escritor = csv.writer(buffer)
    ultimos = {}
    lectura = scanner.Lectura("ABC-12345", "CODE128", 1, 1, 20, 10, "rapido")
    scanner.registrar([lectura], escritor, buffer, ultimos, 5, 2, 100.0)
    exigir(buffer.getvalue() == "", "la primera lectura no se anota si --confirmar 2")
    scanner.registrar([lectura], escritor, buffer, ultimos, 5, 2, 100.4)
    exigir("ABC-12345" in buffer.getvalue(), "la segunda lectura seguida sí se anota")

    print("Dígito de control")
    exigir(scanner.codigo_valido("EAN13", "5901234123457"), "EAN-13 válido")
    exigir(not scanner.codigo_valido("EAN13", "5901234123458"), "EAN-13 inválido")
    exigir(scanner.codigo_valido("EAN8", "55123457"), "EAN-8 válido")
    exigir(not scanner.codigo_valido("EAN8", "55123458"), "EAN-8 inválido")
    exigir(scanner.codigo_valido("UPCA", "012345678905"), "UPC-A válido")
    exigir(scanner.codigo_valido("CODE128", "ABC"), "Code 128 no lleva este dígito")
    try:
        scanner.simbolos_zbar("no-existe")
        exigir(False, "símbolo desconocido debe abortar")
    except SystemExit:
        exigir(True, "símbolo desconocido aborta")

    casos = [
        ("QR 200px", "qr", colocar(fuentes["qr"], 640, 480, 200), "rapido"),
        ("QR 140px", "qr", colocar(fuentes["qr"], 640, 480, 140), "rapido"),
        ("QR poco contraste", "qr",
         colocar(fuentes["qr"], 640, 480, 220, contraste=0.45, brillo=80),
         "rapido"),
        ("QR desenfoque", "qr", colocar(fuentes["qr"], 640, 480, 220, blur=1.2),
         "rapido"),
        ("QR 1280 reducido", "qr",
         escena_optica(fuentes["qr"], 200, (1280, 720)), "rapido"),
        ("EAN ancho", "ean13", colocar(fuentes["ean13"], 640, 480, 320), "rapido"),
        ("Code128 ancho", "code128",
         colocar(fuentes["code128"], 640, 480, 420), "rapido"),
        # Por debajo de ~230 px el Code 128 ya no se lee en gris; el enfoque sí.
        ("Code128 220px", "code128",
         colocar(fuentes["code128"], 640, 480, 220), None),
        # A 30° la línea de zbar no cruza bien las barras; un giro de rescate sí.
        ("Code128 280px 30°", "code128",
         colocar(fuentes["code128"], 640, 480, 280, angulo=30), None),
        # Desenfoque leve: hace falta ampliar el gris enfocado.
        ("Code128 360px blur", "code128",
         colocar(fuentes["code128"], 640, 480, 360, blur=1.4), None),
        ("EAN 170px", "ean13", colocar(fuentes["ean13"], 640, 480, 170), None),
        ("EAN ancho sigue rápido", "ean13",
         colocar(fuentes["ean13"], 640, 480, 360), "rapido"),
        ("EAN óptico 150 @640", "ean13",
         escena_optica(fuentes["ean13"], 150, (640, 480), blur=0.4), None),
        ("EAN óptico 150 @1280", "ean13",
         escena_optica(fuentes["ean13"], 150, (1280, 720), blur=0.4), None),
        ("Code128 óptico 180 @640", "code128",
         escena_optica(fuentes["code128"], 180, (640, 480), blur=0.35), None),
        ("Code128 óptico 180 @1280", "code128",
         escena_optica(fuentes["code128"], 180, (1280, 720), blur=0.35), None),
    ]

    print("\nCasos")
    print(f"{'caso':28} {'base':6} {'nuevo':6} vía")
    vistos_1d_recuperados = 0
    qr_base_ok = 0
    qr_nuevo_ok = 0
    resultado = {}
    for titulo, clave, imagen, via_esperada in casos:
        esperado = ESPERADO[clave]
        base, lecturas = evaluar(imagen, simbolos)
        textos = {lectura.texto for lectura in lecturas}
        vias = ",".join(sorted({lectura.via for lectura in lecturas})) or "-"
        base_ok = esperado in base and not any(t != esperado for t in base)
        nuevo_ok = textos == {esperado}
        marca_base = "si" if base_ok else ("mal" if base else "no")
        marca_nueva = "si" if nuevo_ok else ("mal" if textos else "no")
        print(f"{titulo:28} {marca_base:6} {marca_nueva:6} {vias}")
        resultado[titulo] = (base_ok, nuevo_ok, vias)
        if textos - {esperado}:
            fallos.append(f"{titulo} leyó de más: {textos}")
        if clave == "qr":
            if base_ok:
                qr_base_ok += 1
            if nuevo_ok:
                qr_nuevo_ok += 1
            if base_ok and not nuevo_ok:
                fallos.append(f"regresión QR: {titulo}")
            if via_esperada == "rapido" and nuevo_ok and vias != "rapido":
                fallos.append(f"{titulo} no salió por la vía rápida ({vias})")
        else:
            if not base_ok and nuevo_ok:
                vistos_1d_recuperados += 1
            if via_esperada == "rapido" and nuevo_ok and vias != "rapido":
                fallos.append(f"{titulo} no salió por la vía rápida ({vias})")
            if via_esperada == "rapido" and not nuevo_ok:
                fallos.append(f"{titulo} debería leerse en la vía rápida")

    exigir(qr_nuevo_ok == qr_base_ok and qr_base_ok >= 4,
           f"QR conservados ({qr_nuevo_ok}/{qr_base_ok} de los que la base leía)")
    exigir(vistos_1d_recuperados >= 3,
           f"al menos 3 códigos 1D recuperados ({vistos_1d_recuperados})")
    # Misma escena física: a 640 no se lee; a 1280 el rescate de resolución sí.
    exigir(
        not resultado["Code128 óptico 180 @640"][0]
        and not resultado["Code128 óptico 180 @640"][1]
        and resultado["Code128 óptico 180 @1280"][1]
        and resultado["Code128 óptico 180 @1280"][2] == "resolucion",
        "Code128 pequeño solo aparece al capturar a 1280 (vía resolucion)",
    )

    print("\nCaja del rescate")

    def tinta(imagen):
        ys, xs = np.where(imagen < 180)
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    def toca_tinta(lectura, zona, margen=16):
        x0, y0, x1, y1 = zona
        return (
            lectura.x + lectura.w >= x0 - margen
            and lectura.x <= x1 + margen
            and lectura.y + lectura.h >= y0 - margen
            and lectura.y <= y1 + margen
        )

    for titulo, imagen, via_prefijo in (
        ("giro", colocar(fuentes["code128"], 640, 480, 280, angulo=30), "giro"),
        ("ampliar", colocar(fuentes["code128"], 640, 480, 360, blur=1.4), "ampliar"),
    ):
        lecturas = scanner.leer_hasta_encontrar(imagen, simbolos)
        if not lecturas:
            fallos.append(f"no hubo caja que comprobar ({titulo})")
            print(f"  FALLO  no hubo lectura ({titulo})")
            continue
        lectura = lecturas[0]
        exigir(
            lectura.via.startswith(via_prefijo) and toca_tinta(lectura, tinta(imagen)),
            f"{titulo}: caja ({lectura.x},{lectura.y} {lectura.w}x{lectura.h}) "
            f"vía {lectura.via} sobre el código",
        )

    imagen_mapa, mapa = scanner._variante(
        np.zeros((48, 64), np.uint8), "ampliar"
    )
    exigir(
        callable(mapa) and imagen_mapa.shape == (96, 128) and mapa(20, 10) == (10.0, 5.0),
        "el mapa de ampliar vuelve a la escala original",
    )

    print("\nNombres de rescate")
    exigir(
        "resolucion" not in scanner.nombres_rescate(640, 480, 640),
        "a 640 no se pide resolución nativa",
    )
    exigir(
        scanner.nombres_rescate(1280, 720, 640)[0] == "resolucion",
        "a 1280 el primer rescate es la resolución nativa",
    )
    exigir(
        "ampliar" not in scanner.nombres_rescate(1280, 720, 640),
        "a 1280 no se amplía otra vez",
    )

    print("\nFoto por la línea de comandos")
    foto = colocar(fuentes["qr"], 640, 480, 200)
    with tempfile.TemporaryDirectory() as tmp:
        ruta = os.path.join(tmp, "foto.png")
        csv_ruta = os.path.join(tmp, "registro.csv")
        cv2.imwrite(ruta, foto)
        proc = subprocess.run(
            [sys.executable, os.path.join(RAIZ, "scanner.py"),
             "--imagen", ruta, "--csv", csv_ruta, "--sin-rescate"],
            check=False,
            capture_output=True,
            text=True,
        )
        exigir(proc.returncode == 0, f"scanner.py --imagen sale 0 ({proc.returncode})")
        exigir(
            ESPERADO["qr"] in proc.stdout,
            "la foto QR se imprime",
        )
        exigir(os.path.exists(csv_ruta) and ESPERADO["qr"] in open(csv_ruta, encoding="utf-8").read(),
               "el CSV de la foto contiene el QR")
        dificil = colocar(fuentes["code128"], 640, 480, 220)
        ruta_d = os.path.join(tmp, "code128.png")
        cv2.imwrite(ruta_d, dificil)
        proc_d = subprocess.run(
            [sys.executable, os.path.join(RAIZ, "scanner.py"),
             "--imagen", ruta_d, "--csv", os.path.join(tmp, "c128.csv")],
            check=False, capture_output=True, text=True,
        )
        exigir(
            proc_d.returncode == 0 and "CODE128" in proc_d.stdout and "enfoque" in proc_d.stdout,
            "una foto Code128 justa se lee por enfoque",
        )

    print()
    if fallos:
        print(f"{len(fallos)} fallo(s):")
        for fallo in fallos:
            print(f"- {fallo}")
        return 1
    print("Detección 1D mejorada y QR en la vía rápida.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
