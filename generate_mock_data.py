"""Genera los datos mockeados de GenPet de forma determinística (semilla fija).

Salidas:
- data/seed/{productos,clientes,pedidos}.json -> los carga setup_db.py en la base SQLite de pedidos.
- data/docs/historial_ventas_2026.md y data/docs/resultados_financieros_2026.md -> los indexa ingest.py en el RAG.

Los reportes se calculan a partir de los mismos pedidos, así los números del RAG y de las tools
de pedidos son coherentes entre sí. Uso: python generate_mock_data.py
"""
import json
import random
from collections import defaultdict
from datetime import date, timedelta
from typing import Dict, List

from config import DOCS_DIR, SEED_DIR
from formatting import ars

SEMILLA = 42
ANIO = 2026
MESES = list(range(1, 10))  # enero a septiembre 2026 (cierre del Q3)
NOMBRE_MES = {1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
              7: "Julio", 8: "Agosto", 9: "Septiembre"}

# Precios de lista y costos unitarios (ARS). En julio hubo un aumento de lista del 8%.
PRODUCTOS = [
    {"producto_id": "GP-20", "nombre": "Paños higiénicos GenPet 60x90 cm - Pack x20", "unidades_por_pack": 20,
     "medida": "60x90 cm", "precio_lista": 9800, "precio_lista_desde_julio": 10600, "costo_unitario": 5600,
     "stock_packs": 1240},
    {"producto_id": "GP-50", "nombre": "Paños higiénicos GenPet 60x90 cm - Pack x50", "unidades_por_pack": 50,
     "medida": "60x90 cm", "precio_lista": 21900, "precio_lista_desde_julio": 23650, "costo_unitario": 12300,
     "stock_packs": 610},
]

# Descuento comercial por tipo de cliente (ver politicas_comerciales.md).
DESCUENTO_POR_TIPO = {"distribuidor": 15, "pet shop": 10, "veterinaria": 5, "consumidor final": 0}

CLIENTES = [
    {"cliente_id": 101, "nombre": "Patitas Felices", "tipo": "pet shop", "ciudad": "Córdoba"},
    {"cliente_id": 102, "nombre": "Veterinaria San Roque", "tipo": "veterinaria", "ciudad": "Rosario"},
    {"cliente_id": 103, "nombre": "Distribuidora Mascotas del Sur", "tipo": "distribuidor", "ciudad": "Bahía Blanca"},
    {"cliente_id": 104, "nombre": "Huellitas Pet Shop", "tipo": "pet shop", "ciudad": "CABA"},
    {"cliente_id": 105, "nombre": "Clínica Veterinaria Norte", "tipo": "veterinaria", "ciudad": "Tucumán"},
    {"cliente_id": 106, "nombre": "Laura Gómez", "tipo": "consumidor final", "ciudad": "La Plata"},
    {"cliente_id": 107, "nombre": "Mundo Animal Mayorista", "tipo": "distribuidor", "ciudad": "Mendoza"},
    {"cliente_id": 108, "nombre": "Refugio Cola Contenta", "tipo": "consumidor final", "ciudad": "Mar del Plata"},
    {"cliente_id": 109, "nombre": "Pet Center Palermo", "tipo": "pet shop", "ciudad": "CABA"},
    {"cliente_id": 110, "nombre": "Veterinaria Los Andes", "tipo": "veterinaria", "ciudad": "San Juan"},
    {"cliente_id": 111, "nombre": "Martín Ruiz", "tipo": "consumidor final", "ciudad": "Neuquén"},
    {"cliente_id": 112, "nombre": "Guardería Canina Dog House", "tipo": "pet shop", "ciudad": "Salta"},
]

# Packs por pedido según el tipo de cliente: (mín, máx) para cada producto.
VOLUMEN_POR_TIPO = {
    "distribuidor": {"GP-20": (10, 40), "GP-50": (8, 30)},
    "pet shop": {"GP-20": (4, 15), "GP-50": (2, 10)},
    "veterinaria": {"GP-20": (2, 8), "GP-50": (1, 6)},
    "consumidor final": {"GP-20": (1, 3), "GP-50": (0, 2)},
}
CANAL_POR_TIPO = {"distribuidor": "mayorista", "pet shop": "mayorista",
                  "veterinaria": "whatsapp", "consumidor final": "web"}
# Pedidos por mes: el negocio crece a lo largo del año.
PEDIDOS_POR_MES = {1: 12, 2: 11, 3: 14, 4: 15, 5: 17, 6: 18, 7: 20, 8: 22, 9: 24}
# Peso de cada cliente al sortear quién compra (los distribuidores compran menos seguido pero más volumen).
PESO_POR_TIPO = {"distribuidor": 2, "pet shop": 4, "veterinaria": 3, "consumidor final": 2}

COSTO_LOGISTICO_PCT = 6       # % sobre ingresos netos (envíos y embalaje)
GASTOS_FIJOS_MENSUALES = 400_000  # alquiler del depósito, servicios y marketing digital


def precio_vigente(producto: dict, fecha: date) -> int:
    return producto["precio_lista_desde_julio"] if fecha.month >= 7 else producto["precio_lista"]


def generar_pedidos(rng: random.Random) -> List[dict]:
    productos = {p["producto_id"]: p for p in PRODUCTOS}
    pesos = [PESO_POR_TIPO[c["tipo"]] for c in CLIENTES]
    pedidos: List[dict] = []
    pedido_id = 5001

    for mes in MESES:
        inicio = date(ANIO, mes, 1)
        dias_mes = ((date(ANIO, mes + 1, 1) if mes < 12 else date(ANIO + 1, 1, 1)) - inicio).days
        fechas = sorted(inicio + timedelta(days=rng.randrange(dias_mes)) for _ in range(PEDIDOS_POR_MES[mes]))

        for fecha in fechas:
            cliente = rng.choices(CLIENTES, weights=pesos)[0]
            volumen = VOLUMEN_POR_TIPO[cliente["tipo"]]
            # El pack x50 gana participación a lo largo del año (campaña "Rinde más").
            prob_x50 = 0.45 + 0.04 * mes
            items = []
            for producto_id in ("GP-20", "GP-50"):
                if producto_id == "GP-50" and rng.random() > prob_x50:
                    continue
                cantidad = rng.randint(*volumen[producto_id])
                if cantidad:
                    items.append({"producto_id": producto_id, "cantidad": cantidad,
                                  "precio_unitario": precio_vigente(productos[producto_id], fecha)})
            if not items:
                items.append({"producto_id": "GP-20", "cantidad": 1,
                              "precio_unitario": precio_vigente(productos["GP-20"], fecha)})

            descuento = DESCUENTO_POR_TIPO[cliente["tipo"]]
            subtotal = sum(i["cantidad"] * i["precio_unitario"] for i in items)
            total = round(subtotal * (100 - descuento) / 100)

            # Septiembre tiene pedidos todavía en curso; el resto del año, alguna cancelación ocasional.
            if fecha >= date(ANIO, 9, 24):
                estado = rng.choice(["en preparación", "despachado"])
            else:
                estado = "cancelado" if rng.random() < 0.04 else "entregado"

            pedidos.append({
                "pedido_id": pedido_id, "cliente_id": cliente["cliente_id"], "fecha": fecha.isoformat(),
                "canal": CANAL_POR_TIPO[cliente["tipo"]], "estado": estado, "descuento_pct": descuento,
                "subtotal": subtotal, "total": total, "items": items,
            })
            pedido_id += 1
    return pedidos


def resumen_mensual(pedidos: List[dict]) -> Dict[int, dict]:
    costos = {p["producto_id"]: p["costo_unitario"] for p in PRODUCTOS}
    resumen: Dict[int, dict] = defaultdict(lambda: {"pedidos": 0, "cancelados": 0, "packs": defaultdict(int),
                                                    "ingresos": 0, "ingresos_por_producto": defaultdict(float),
                                                    "ingresos_por_canal": defaultdict(int), "costo_mercaderia": 0})
    for pedido in pedidos:
        mes = int(pedido["fecha"][5:7])
        r = resumen[mes]
        if pedido["estado"] == "cancelado":
            r["cancelados"] += 1
            continue
        r["pedidos"] += 1
        r["ingresos"] += pedido["total"]
        r["ingresos_por_canal"][pedido["canal"]] += pedido["total"]
        factor = (100 - pedido["descuento_pct"]) / 100
        for item in pedido["items"]:
            r["packs"][item["producto_id"]] += item["cantidad"]
            r["ingresos_por_producto"][item["producto_id"]] += item["cantidad"] * item["precio_unitario"] * factor
            r["costo_mercaderia"] += item["cantidad"] * costos[item["producto_id"]]
    for r in resumen.values():
        r["logistica"] = r["ingresos"] * COSTO_LOGISTICO_PCT / 100
        r["margen_bruto"] = r["ingresos"] - r["costo_mercaderia"] - r["logistica"]
        r["gastos_fijos"] = GASTOS_FIJOS_MENSUALES
        r["ganancia_neta"] = r["margen_bruto"] - GASTOS_FIJOS_MENSUALES
    return resumen


def trimestre(meses: List[int], resumen: Dict[int, dict], campo: str) -> float:
    return sum(resumen[m][campo] for m in meses)


TRIMESTRES = {"Q1": [1, 2, 3], "Q2": [4, 5, 6], "Q3": [7, 8, 9]}


def render_historial_ventas(resumen: Dict[int, dict], pedidos: List[dict]) -> str:
    nombres = {c["cliente_id"]: c["nombre"] for c in CLIENTES}
    lineas = [
        "# GenPet · Historial de ventas 2026 (enero a septiembre)",
        "",
        "Reporte comercial generado a partir del sistema de pedidos de GenPet. Incluye solo pedidos no cancelados.",
        "Montos en pesos argentinos (ARS), netos de descuentos comerciales.",
        "",
        "## Ventas mensuales",
        "",
        "| Mes | Pedidos | Packs x20 vendidos | Packs x50 vendidos | Ingresos netos | Pedidos cancelados |",
        "|---|---|---|---|---|---|",
    ]
    for mes in MESES:
        r = resumen[mes]
        lineas.append(f"| {NOMBRE_MES[mes]} | {r['pedidos']} | {r['packs']['GP-20']} | {r['packs']['GP-50']} | "
                      f"{ars(r['ingresos'])} | {r['cancelados']} |")

    lineas += ["", "## Ventas por trimestre", "",
               "| Trimestre | Pedidos | Packs x20 | Packs x50 | Ingresos netos |", "|---|---|---|---|---|"]
    for nombre, meses in TRIMESTRES.items():
        lineas.append(
            f"| {nombre} {ANIO} | {sum(resumen[m]['pedidos'] for m in meses)} | "
            f"{sum(resumen[m]['packs']['GP-20'] for m in meses)} | {sum(resumen[m]['packs']['GP-50'] for m in meses)} | "
            f"{ars(trimestre(meses, resumen, 'ingresos'))} |")

    total_x20 = sum(resumen[m]["packs"]["GP-20"] for m in MESES)
    total_x50 = sum(resumen[m]["packs"]["GP-50"] for m in MESES)
    ing_x20 = sum(resumen[m]["ingresos_por_producto"]["GP-20"] for m in MESES)
    ing_x50 = sum(resumen[m]["ingresos_por_producto"]["GP-50"] for m in MESES)
    ingresos = sum(resumen[m]["ingresos"] for m in MESES)
    lineas += [
        "", "## Ventas por producto (acumulado enero-septiembre)", "",
        "| Producto | Packs vendidos | Paños vendidos | Ingresos netos | Participación en ingresos |",
        "|---|---|---|---|---|",
        f"| Pack x20 (GP-20) | {total_x20} | {total_x20 * 20} | {ars(ing_x20)} | {ing_x20 / ingresos:.0%} |",
        f"| Pack x50 (GP-50) | {total_x50} | {total_x50 * 50} | {ars(ing_x50)} | {ing_x50 / ingresos:.0%} |",
        "",
        "El pack x20 es el más vendido en cantidad de packs, pero el pack x50 genera la mayor parte de los "
        "ingresos y gana participación mes a mes desde la campaña \"Rinde más\" lanzada en abril.",
    ]

    lineas += ["", "## Ventas por canal (acumulado enero-septiembre)", "",
               "| Canal | Ingresos netos | Participación |", "|---|---|---|"]
    for canal in ("mayorista", "whatsapp", "web"):
        monto = sum(resumen[m]["ingresos_por_canal"][canal] for m in MESES)
        lineas.append(f"| {canal} | {ars(monto)} | {monto / ingresos:.0%} |")

    por_cliente: Dict[int, int] = defaultdict(int)
    for p in pedidos:
        if p["estado"] != "cancelado":
            por_cliente[p["cliente_id"]] += p["total"]
    lineas += ["", "## Top 5 clientes por facturación (acumulado enero-septiembre)", "",
               "| Posición | Cliente | ID | Facturación neta |", "|---|---|---|---|"]
    top = sorted(por_cliente.items(), key=lambda kv: kv[1], reverse=True)[:5]
    for pos, (cliente_id, monto) in enumerate(top, start=1):
        lineas.append(f"| {pos} | {nombres[cliente_id]} | {cliente_id} | {ars(monto)} |")
    return "\n".join(lineas) + "\n"


def render_resultados(resumen: Dict[int, dict]) -> str:
    lineas = [
        "# GenPet · Resultados financieros 2026 (ganancias y costos)",
        "",
        "Estado de resultados simplificado de GenPet, elaborado por administración. Montos en pesos argentinos (ARS).",
        "",
        "Supuestos del cálculo:",
        "",
        "- **Ingresos netos:** facturación de pedidos no cancelados, neta de descuentos comerciales.",
        "- **Costo de mercadería:** costo unitario de fabricación por pack (x20: $5.600; x50: $12.300).",
        f"- **Costo logístico:** {COSTO_LOGISTICO_PCT}% de los ingresos netos (envíos y embalaje).",
        f"- **Gastos fijos:** {ars(GASTOS_FIJOS_MENSUALES)} por mes (alquiler del depósito, servicios y "
        "marketing digital).",
        "",
        "## Resultado mensual",
        "",
        "| Mes | Ingresos netos | Costo de mercadería | Costo logístico | Margen bruto | Gastos fijos | Ganancia neta |",
        "|---|---|---|---|---|---|---|",
    ]
    for mes in MESES:
        r = resumen[mes]
        lineas.append(f"| {NOMBRE_MES[mes]} | {ars(r['ingresos'])} | {ars(r['costo_mercaderia'])} | "
                      f"{ars(r['logistica'])} | {ars(r['margen_bruto'])} | {ars(r['gastos_fijos'])} | "
                      f"{ars(r['ganancia_neta'])} |")

    lineas += ["", "## Resultado por trimestre", "",
               "| Trimestre | Ingresos netos | Margen bruto | Ganancia neta | Margen neto |", "|---|---|---|---|---|"]
    previo = None
    comentarios = []
    for nombre, meses in TRIMESTRES.items():
        ingresos = trimestre(meses, resumen, "ingresos")
        ganancia = trimestre(meses, resumen, "ganancia_neta")
        lineas.append(f"| {nombre} {ANIO} | {ars(ingresos)} | {ars(trimestre(meses, resumen, 'margen_bruto'))} | "
                      f"{ars(ganancia)} | {ganancia / ingresos:.1%} |")
        if previo:
            variacion = (ganancia - previo[1]) / abs(previo[1])
            comentarios.append(f"- La ganancia neta pasó de {ars(previo[1])} en el {previo[0]} a {ars(ganancia)} "
                               f"en el {nombre} ({variacion:+.0%}).")
        previo = (nombre, ganancia)

    total_ing = sum(resumen[m]["ingresos"] for m in MESES)
    total_gan = sum(resumen[m]["ganancia_neta"] for m in MESES)
    mejor = max(MESES, key=lambda m: resumen[m]["ganancia_neta"])
    lineas += [
        "", "## Análisis de administración", "",
        *comentarios,
        f"- Acumulado enero-septiembre: ingresos netos de {ars(total_ing)} y ganancia neta de {ars(total_gan)} "
        f"(margen neto {total_gan / total_ing:.1%}).",
        f"- El mejor mes del año fue {NOMBRE_MES[mejor]} con una ganancia neta de {ars(resumen[mejor]['ganancia_neta'])}.",
        "- El aumento de lista del 8% aplicado en julio mejoró el margen bruto sin afectar el volumen de pedidos.",
        "- Los meses de verano (enero y febrero) muestran menor actividad: los gastos fijos pesan más sobre el resultado.",
    ]
    return "\n".join(lineas) + "\n"


def main() -> None:
    rng = random.Random(SEMILLA)
    pedidos = generar_pedidos(rng)
    resumen = resumen_mensual(pedidos)

    SEED_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for nombre, datos in (("productos", PRODUCTOS), ("clientes", CLIENTES), ("pedidos", pedidos)):
        (SEED_DIR / f"{nombre}.json").write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    (DOCS_DIR / "historial_ventas_2026.md").write_text(render_historial_ventas(resumen, pedidos), encoding="utf-8")
    (DOCS_DIR / "resultados_financieros_2026.md").write_text(render_resultados(resumen), encoding="utf-8")
    print(f"Generados {len(CLIENTES)} clientes, {len(pedidos)} pedidos y 2 reportes en {DOCS_DIR}")


if __name__ == "__main__":
    main()
