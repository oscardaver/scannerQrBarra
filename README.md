# Escáner QR y códigos de barras (Raspberry Pi 3)

Lee códigos con la cámara, con OpenCV y pyzbar, y anota fecha, hora, tipo y texto en `registro_codigos.csv`.

## Por qué el QR iba bien y el código de barras no

No era un solo fallo. En imágenes de prueba (EAN-13, Code 128 y QR, con poca resolución, desenfoque e inclinación) se vio esto:

1. **Píxeles por barra.** Un EAN-13 son 95 módulos; un Code 128 del mismo ancho impreso tiene las barras más finas. Un QR del mismo tamaño tiene módulos más grandes y corrección de errores. A 640×480, un código que ocupa poca parte del encuadre cae cerca de 1 px por barra y zbar deja de ver los bordes. El QR sigue leyéndose. Pedir 1280×720 a la cámara (y, si hace falta, decodificar ese fotograma entero) es lo que recupera el detalle. Ampliar por software un 640 ya borroso no separa barras que el sensor fundió.

2. **Inclinación y desenfoque.** ZBar recorre el fotograma en líneas horizontales y verticales, no en diagonal. Un Code 128 girado ~25° falla donde el QR no. Un desenfoque leve hace lo mismo con las barras finas. Además, con la densidad por defecto (1, o sea todas las filas), zbar **descarta** lecturas EAN de calidad menor que 4. El QR no pasa por ese filtro. La densidad no estaba “apagada”; subirla no era la solución.

3. **Preprocesar siempre empeora el QR.** Umbral de Otsu, umbral adaptativo y CLAHE hicieron fallar QR de poco contraste que el gris simple leía. Por eso el fotograma se sigue leyendo primero en gris. Solo si esa lectura sale vacía se prueba **un** rescate (enfoque, ×2, o un giro). El QR nítido no entra en ese camino.

4. **Saltar fotogramas.** Analizar 1 de cada 3 ahorra CPU, pero tira imágenes. El desenfoque de movimiento borra un código 1D y casi no afecta al QR. El salto sigue en 3 para no cargar la Pi; con el código en la mano conviene `--saltar 1`.

## Qué cambió

- La cámara se pide a **1280×720**. La vía rápida reduce el gris a 640 px del lado largo, así el QR cuesta parecido a antes. Si no hay lectura, el rescate usa el fotograma completo.
- Si la cámara no acepta ese tamaño o no entrega imagen, se reintenta a **640×480** y se avisa.
- En 640×480 el rescate enfoca, amplía ×2 o gira (±15° y ±30°). Como mucho hay dos decodificaciones por fotograma analizado.
- Símbolos por defecto: QR, EAN-8/13, UPC-A/E, Code 128/39/93 y Codabar. Interleaved 2 of 5 queda fuera porque inventa lecturas cortas; `--simbolos todos` lo vuelve a activar.
- Se comprueba el dígito de control de EAN-13, EAN-8 y UPC-A.
- La ventana de vista previa se muestra a 800 px de ancho. `--headless` la quita.

## Probar en la Raspberry Pi

Dependencias: `python3-opencv`, `libzbar0`, `pyzbar`.

```bash
python3 scanner.py --headless
```

Cada 5 segundos imprime capturas/s y análisis/s. Si la Pi se queda corta:

```bash
python3 scanner.py --headless --ancho 640 --alto 480 --saltar 3 --sin-rescate
```

Eso se acerca al comportamiento anterior (solo gris, 640×480, 1 de cada 3).

Para un código de barras pequeño o en movimiento:

```bash
python3 scanner.py --headless --ancho 1280 --alto 720 --saltar 1
```

`--mjpg` pide MJPEG si la USB de la Pi 3 no aguanta 720p sin comprimir. Deja margen blanco alrededor del código, sujétalo quieto y evita el contraluz: el movimiento funde las barras y el QR a veces se lee igual.

Una foto, sin cámara:

```bash
python3 scanner.py --imagen foto.png
```

La línea entre paréntesis (`enfoque`, `ampliar`, `resolucion`, `giro15`, …) indica que no bastó la vía rápida.

## Probar sin cámara

`probar_deteccion.py` arma escenas a partir de `pruebas/*.png` y compara una sola llamada a pyzbar (lo de antes) con el ciclo nuevo. Exige que los QR salgan por la vía rápida y que haya códigos 1D que antes no se leían.

```bash
python3 probar_deteccion.py
```

## Opciones

| Opción | Default | Para qué |
| --- | --- | --- |
| `--camara N` | 0 | Índice de la cámara |
| `--ancho` / `--alto` | 1280 / 720 | Resolución pedida |
| `--saltar N` | 3 | Analiza 1 de cada N fotogramas |
| `--max-lado` | 640 | Tope de la vía rápida |
| `--sin-rescate` | no | Solo el gris, sin enfoque ni giros |
| `--headless` | no | Sin ventana |
| `--mjpg` | no | Pide MJPEG |
| `--simbolos` | QR y 1D habituales | Lista separada por comas, o `todos` |
| `--csv` | `registro_codigos.csv` | Registro |
| `--espera` | 5 | Segundos antes de repetir el mismo texto |
| `--confirmar N` | 1 | Lecturas iguales seguidas antes de anotar |
| `--imagen RUTA` | | Una foto y termina |
