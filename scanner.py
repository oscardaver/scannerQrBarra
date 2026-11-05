#!/usr/bin/env python3
"""
Scanner de códigos de barras y QR para Raspberry Pi 3.
Registra fecha, hora, tipo y texto de cada código en un CSV.

Los QR se leen bien en gris y a poca resolución: tienen módulos grandes
y corrección de errores. Los códigos 1D (EAN-13, Code 128, …) fallan
cuando cada barra ocupa cerca de un píxel, cuando están inclinados o
cuando el desenfoque funde las barras. ZBar, además, tira las lecturas
EAN de calidad baja. Por eso la vía rápida sigue siendo el gris (no
empeora el QR) y, solo si no hubo lectura, se prueba un único rescate
por fotograma: resolución nativa, enfoque, ampliación o un giro.

Dependencias: python3-opencv, libzbar0, pyzbar
"""
import argparse
import csv
import os
import time
from collections import namedtuple
from datetime import datetime

import cv2
import numpy as np
from pyzbar.pyzbar import ZBarSymbol, decode

ARCHIVO_CSV = "registro_codigos.csv"
ANCHO, ALTO = 1280, 720
MAX_LADO_RAPIDO = 640
MAX_LADO_GIRO = 960
SALTAR_FRAMES = 3
ESPERA_REPETIDO = 5
SIMBOLOS_DEFECTO = (
    "qrcode,ean13,ean8,upca,upce,code128,code39,code93,codabar"
)

Lectura = namedtuple("Lectura", "texto tipo x y w h via")

_ALIAS_SIMBOLO = {
    "qr": "qrcode",
    "ean-13": "ean13",
    "ean-8": "ean8",
    "upc-a": "upca",
    "upc-e": "upce",
    "code-128": "code128",
    "code-39": "code39",
    "code-93": "code93",
}


def simbolos_zbar(texto):
    """Lista de ZBarSymbol, o None para dejar el criterio de zbar."""
    if texto is None or texto.strip().lower() in {"", "todos", "all"}:
        return None
    por_nombre = {s.name.lower(): s for s in ZBarSymbol}
    salida = []
    for parte in texto.split(","):
        nombre = parte.strip().lower().replace(" ", "")
        nombre = _ALIAS_SIMBOLO.get(nombre, nombre)
        if nombre not in por_nombre or nombre in {"none", "partial"}:
            conocidos = ", ".join(sorted(
                n for n in por_nombre if n not in {"none", "partial"}
            ))
            raise SystemExit(
                f"Símbolo desconocido: {parte.strip()}. Conocidos: {conocidos}"
            )
        if por_nombre[nombre] not in salida:
            salida.append(por_nombre[nombre])
    return salida


def codigo_valido(tipo, texto):
    """Descarta EAN/UPC-A cuyo dígito de control no cuadra.

    ZBar ya filtra casi todos; esto evita anotar un producto equivocado
    cuando una lectura 1D dudosa se cuela igual.
    """
    if tipo == "EAN13":
        return _digito_ok(texto, 13, empieza_en_tres=False)
    if tipo == "EAN8":
        return _digito_ok(texto, 8, empieza_en_tres=True)
    if tipo == "UPCA":
        return _digito_ok(texto, 12, empieza_en_tres=True)
    return True


def _digito_ok(texto, largo, empieza_en_tres):
    if len(texto) != largo or not texto.isdigit():
        return False
    suma = 0
    for i, caracter in enumerate(texto[:-1]):
        peso = 3 if (i % 2 == 0) == empieza_en_tres else 1
        suma += int(caracter) * peso
    return (10 - suma % 10) % 10 == int(texto[-1])


def a_gris(frame):
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def reducir(gris, max_lado):
    """Devuelve (imagen, (sx, sy)) donde sx/sy llevan coordenadas a `gris`."""
    alto, ancho = gris.shape[:2]
    if max_lado <= 0 or max(alto, ancho) <= max_lado:
        return gris, (1.0, 1.0)
    factor = max_lado / float(max(alto, ancho))
    nuevo = (
        max(1, int(round(ancho * factor))),
        max(1, int(round(alto * factor))),
    )
    pequenio = cv2.resize(gris, nuevo, interpolation=cv2.INTER_AREA)
    return pequenio, (ancho / pequenio.shape[1], alto / pequenio.shape[0])


def enfocar(gris):
    # Unsharp suave: separa barras finas. No se usa en la vía rápida
    # porque en QR de poco contraste puede borrar el código.
    suave = cv2.GaussianBlur(gris, (0, 0), 0.8)
    return cv2.addWeighted(gris, 1.6, suave, -0.6, 0)


def ampliar(gris, factor):
    grande = cv2.resize(
        gris, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR
    )
    return grande, _escala(
        gris.shape[1] / grande.shape[1],
        gris.shape[0] / grande.shape[0],
    )


def rotar(gris, angulo):
    """Gira sin recortar las esquinas. El mapa vuelve al encuadre original."""
    alto, ancho = gris.shape[:2]
    centro = (ancho / 2.0, alto / 2.0)
    matriz = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    coseno = abs(matriz[0, 0])
    seno = abs(matriz[0, 1])
    nuevo_w = max(1, int(round(alto * seno + ancho * coseno)))
    nuevo_h = max(1, int(round(alto * coseno + ancho * seno)))
    matriz[0, 2] += nuevo_w / 2.0 - centro[0]
    matriz[1, 2] += nuevo_h / 2.0 - centro[1]
    girada = cv2.warpAffine(
        gris,
        matriz,
        (nuevo_w, nuevo_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    inversa = cv2.invertAffineTransform(matriz)

    def mapa(x, y):
        return (
            inversa[0, 0] * x + inversa[0, 1] * y + inversa[0, 2],
            inversa[1, 0] * x + inversa[1, 1] * y + inversa[1, 2],
        )

    return girada, mapa


def _escala(sx, sy):
    def mapa(x, y):
        return x * sx, y * sy
    return mapa


def _identidad(x, y):
    return x, y


def _componer(mapa_despues, mapa_antes):
    def mapa(x, y):
        xi, yi = mapa_despues(x, y)
        return mapa_antes(xi, yi)
    return mapa


def nombres_rescate(ancho, alto, max_lado):
    """Un rescate por fotograma, en el orden que más códigos 1D recupera.

    Si la captura es mayor que la vía rápida, lo primero es decodificar
    esos píxeles. Ampliar un 1280×720 no aporta detalle y en la Pi 3 sale caro.
    """
    if max_lado > 0 and max(ancho, alto) > max_lado:
        return (
            "resolucion",
            "enfoque",
            "giro15",
            "giro-15",
            "giro30",
            "giro-30",
        )
    return (
        "enfoque",
        "ampliar",
        "giro15",
        "giro-15",
        "giro30",
        "giro-30",
    )


def _variante(gris, nombre):
    if nombre == "resolucion":
        return gris, _identidad
    if nombre == "enfoque":
        return enfocar(gris), _identidad
    if nombre == "ampliar":
        if max(gris.shape) > 1000:
            return enfocar(gris), _identidad
        return ampliar(enfocar(gris), 2.0)
    if nombre.startswith("giro"):
        angulo = int(nombre.replace("giro", ""))
        base, escala = reducir(gris, MAX_LADO_GIRO)
        girada, mapa_giro = rotar(enfocar(base), angulo)
        return girada, _componer(mapa_giro, _escala(*escala))
    raise ValueError(f"Rescate desconocido: {nombre}")


def _leer_simbolos(imagen, simbolos):
    imagen = np.ascontiguousarray(imagen)
    if imagen.dtype != np.uint8:
        imagen = np.clip(imagen, 0, 255).astype(np.uint8)
    try:
        if simbolos:
            crudos = decode(imagen, symbols=simbolos)
        else:
            crudos = decode(imagen)
    except Exception:
        return []
    salida = []
    for codigo in crudos:
        texto = codigo.data.decode("utf-8", errors="replace").strip("\x00").strip()
        tipo = codigo.type if isinstance(codigo.type, str) else str(codigo.type)
        if not texto or not codigo_valido(tipo, texto):
            continue
        salida.append((texto, tipo, codigo.rect))
    return salida


def _empaquetar(crudos, mapa, via):
    lecturas = []
    for texto, tipo, rect in crudos:
        puntos = (
            (rect.left, rect.top),
            (rect.left + rect.width, rect.top),
            (rect.left + rect.width, rect.top + rect.height),
            (rect.left, rect.top + rect.height),
        )
        xs, ys = [], []
        for x, y in puntos:
            xo, yo = mapa(float(x), float(y))
            xs.append(xo)
            ys.append(yo)
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        lecturas.append(Lectura(
            texto,
            tipo,
            int(round(x1)),
            int(round(y1)),
            max(1, int(round(x2 - x1))),
            max(1, int(round(y2 - y1))),
            via,
        ))
    return lecturas


def _deduplicar(lecturas):
    mejores = {}
    for lectura in lecturas:
        clave = (lectura.tipo, lectura.texto)
        previa = mejores.get(clave)
        if previa is None or lectura.w * lectura.h > previa.w * previa.h:
            mejores[clave] = lectura
    return sorted(mejores.values(), key=lambda lectura: (lectura.y, lectura.x, lectura.texto))


def buscar(gris, simbolos=None, fase=0, max_lado=MAX_LADO_RAPIDO, rescate=True):
    """Lee un fotograma.

    Siempre prueba el gris (reducido si la captura es grande). Solo si no
    hay nada y `rescate` está activo, prueba la variante `fase`. Así un QR
    nítido no paga el coste extra ni pasa por un filtro que pueda borrarlo.
    """
    gris = a_gris(gris)
    reducido, escala = reducir(gris, max_lado)
    lecturas = _empaquetar(
        _leer_simbolos(reducido, simbolos), _escala(*escala), "rapido"
    )
    if lecturas or not rescate:
        return _deduplicar(lecturas)
    nombres = nombres_rescate(gris.shape[1], gris.shape[0], max_lado)
    nombre = nombres[fase % len(nombres)]
    imagen, mapa = _variante(gris, nombre)
    return _deduplicar(_empaquetar(_leer_simbolos(imagen, simbolos), mapa, nombre))


def leer_hasta_encontrar(gris, simbolos=None, max_lado=MAX_LADO_RAPIDO, rescate=True):
    """Prueba la vía rápida y, si falla, cada rescate. Para una foto o un código quieto."""
    gris = a_gris(gris)
    lecturas = buscar(gris, simbolos, max_lado=max_lado, rescate=False)
    if lecturas or not rescate:
        return lecturas
    for nombre in nombres_rescate(gris.shape[1], gris.shape[0], max_lado):
        imagen, mapa = _variante(gris, nombre)
        lecturas = _deduplicar(
            _empaquetar(_leer_simbolos(imagen, simbolos), mapa, nombre)
        )
        if lecturas:
            return lecturas
    return []


def abrir_csv(ruta):
    nuevo = not os.path.exists(ruta)
    archivo = open(ruta, "a", newline="", encoding="utf-8")
    escritor = csv.writer(archivo)
    if nuevo:
        escritor.writerow(["fecha", "hora", "tipo", "texto"])
        archivo.flush()
    return archivo, escritor


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


def mostrar(frame):
    alto, ancho = frame.shape[:2]
    vista = frame
    if ancho > 800:
        nuevo_w = 800
        nuevo_h = max(1, int(round(alto * nuevo_w / ancho)))
        vista = cv2.resize(frame, (nuevo_w, nuevo_h), interpolation=cv2.INTER_AREA)
    cv2.imshow("Scanner", vista)
    return (cv2.waitKey(1) & 0xFF) == ord("q")


def dibujar(frame, lectura):
    alto, ancho = frame.shape[:2]
    x, y, w, h = lectura.x, lectura.y, lectura.w, lectura.h
    # ZBar a veces devuelve un solo punto (el arranque de la barra).
    if w < 8 or h < 8:
        cx = x + w / 2.0
        cy = y + h / 2.0
        x, y, w, h = int(cx) - 18, int(cy) - 18, 36, 36
    x1 = max(0, min(ancho - 1, x))
    y1 = max(0, min(alto - 1, y))
    x2 = max(0, min(ancho - 1, x + w))
    y2 = max(0, min(alto - 1, y + h))
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        lectura.texto,
        (x1, max(y1 - 8, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )


def registrar(lecturas, escritor, archivo, ultimos, espera, confirmar, ahora):
    vistos = set()
    for lectura in lecturas:
        if lectura.texto in vistos:
            continue
        vistos.add(lectura.texto)
        if ahora - ultimos.get(lectura.texto, 0) < espera:
            continue
        if confirmar > 1:
            cuenta, desde = ultimos.get("_c:" + lectura.texto, (0, ahora))
            if ahora - desde > 2.0:
                cuenta = 0
            cuenta += 1
            ultimos["_c:" + lectura.texto] = (cuenta, ahora if cuenta == 1 else desde)
            if cuenta < confirmar:
                continue
            ultimos["_c:" + lectura.texto] = (0, ahora)
        ultimos[lectura.texto] = ahora
        momento = datetime.now()
        escritor.writerow([
            momento.strftime("%Y-%m-%d"),
            momento.strftime("%H:%M:%S"),
            lectura.tipo,
            lectura.texto,
        ])
        archivo.flush()
        extra = f" ({lectura.via})" if lectura.via != "rapido" else ""
        print(f"[{momento:%H:%M:%S}] {lectura.tipo}: {lectura.texto}{extra}")


def procesar_foto(ruta, args):
    frame = cv2.imread(ruta, cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"No se pudo leer la imagen: {ruta}")
    simbolos = simbolos_zbar(args.simbolos)
    lecturas = leer_hasta_encontrar(
        frame,
        simbolos,
        max_lado=args.max_lado,
        rescate=not args.sin_rescate,
    )
    if not lecturas:
        print("No se leyó ningún código.")
        return 1
    archivo, escritor = abrir_csv(args.csv)
    try:
        registrar(
            lecturas, escritor, archivo, {}, args.espera, 1, time.time()
        )
    finally:
        archivo.close()
    return 0


def bucle(args):
    simbolos = simbolos_zbar(args.simbolos)
    cam, ancho_real, alto_real = abrir_camara(
        args.camara, args.ancho, args.alto, args.mjpg
    )
    if ancho_real < args.ancho * 0.9 or alto_real < args.alto * 0.9:
        print(
            f"Aviso: la cámara entregó {ancho_real}x{alto_real} "
            f"(se pidió {args.ancho}x{args.alto}). "
            "Un sensor que no llega a 720p deja los códigos 1D pequeños sin detalle."
        )
    else:
        print(f"Cámara {ancho_real}x{alto_real}.")
    print(
        f"Analiza 1 de cada {args.saltar} fotogramas. "
        f"Rescate {'activo' if not args.sin_rescate else 'apagado'}. "
        f"Vía rápida hasta {args.max_lado} px. "
        "Ctrl+C o 'q' para salir."
    )

    archivo, escritor = abrir_csv(args.csv)
    ultimos = {}
    contador = 0
    fase = 0
    headless = args.headless
    inicio = time.time()
    capturas = 0
    analisis = 0
    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            capturas += 1
            contador += 1
            if contador % args.saltar == 0:
                analisis += 1
                lecturas = buscar(
                    frame,
                    simbolos,
                    fase=fase,
                    max_lado=args.max_lado,
                    rescate=not args.sin_rescate,
                )
                if not args.sin_rescate and (
                    not lecturas or any(lectura.via != "rapido" for lectura in lecturas)
                ):
                    fase += 1
                ahora = time.time()
                registrar(
                    lecturas,
                    escritor,
                    archivo,
                    ultimos,
                    args.espera,
                    args.confirmar,
                    ahora,
                )
                if not headless:
                    for lectura in lecturas:
                        dibujar(frame, lectura)

            if not headless:
                try:
                    if mostrar(frame):
                        break
                except cv2.error:
                    print("No hay ventana; sigo sin imagen.")
                    headless = True

            if time.time() - inicio >= 5:
                elapsed = time.time() - inicio
                print(
                    f"ritmo {capturas / elapsed:.1f} capturas/s, "
                    f"{analisis / elapsed:.1f} análisis/s "
                    f"({ancho_real}x{alto_real})"
                )
                inicio = time.time()
                capturas = 0
                analisis = 0
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        archivo.close()
        cv2.destroyAllWindows()
        print("Finalizado.")


def main():
    ap = argparse.ArgumentParser(
        description="Lee QR y códigos de barras 1D y los anota en un CSV."
    )
    ap.add_argument("--headless", action="store_true",
                    help="no mostrar ventana (ahorra CPU en la Pi)")
    ap.add_argument("--camara", type=int, default=0, help="índice de la cámara")
    ap.add_argument("--ancho", type=int, default=ANCHO,
                    help=f"ancho pedido a la cámara (default {ANCHO})")
    ap.add_argument("--alto", type=int, default=ALTO,
                    help=f"alto pedido a la cámara (default {ALTO})")
    ap.add_argument("--saltar", type=int, default=SALTAR_FRAMES,
                    help="analiza 1 de cada N fotogramas (1 = todos; default 3)")
    ap.add_argument("--max-lado", type=int, default=MAX_LADO_RAPIDO,
                    help="lado máximo de la vía rápida, en píxeles (default 640)")
    ap.add_argument("--sin-rescate", action="store_true",
                    help="solo la vía rápida, sin enfoque, ampliación ni giros")
    ap.add_argument("--mjpg", action="store_true",
                    help="pide MJPEG a la cámara (menos USB; a veces más compresión)")
    ap.add_argument("--simbolos", default=SIMBOLOS_DEFECTO,
                    help="lista separada por comas, o 'todos' "
                         f"(default {SIMBOLOS_DEFECTO})")
    ap.add_argument("--csv", default=ARCHIVO_CSV, help="ruta del registro")
    ap.add_argument("--espera", type=float, default=ESPERA_REPETIDO,
                    help="segundos antes de volver a anotar el mismo texto")
    ap.add_argument("--confirmar", type=int, default=1,
                    help="lecturas iguales seguidas antes de anotar (default 1)")
    ap.add_argument("--imagen",
                    help="decodifica una foto y termina, sin abrir la cámara")
    args = ap.parse_args()
    if args.saltar < 1:
        ap.error("--saltar debe ser >= 1")
    if args.confirmar < 1:
        ap.error("--confirmar debe ser >= 1")
    if args.max_lado < 0:
        ap.error("--max-lado debe ser >= 0")
    if args.imagen:
        raise SystemExit(procesar_foto(args.imagen, args))
    bucle(args)


if __name__ == "__main__":
    main()
