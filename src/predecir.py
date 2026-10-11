"""
Paso 3: usar el modelo entrenado para predecir el nivel de congestión.

Ejemplos:
    python src/predecir.py --estacion "Calle 72" --fecha 2024-04-29 --hora 7
    python src/predecir.py --estacion heroes --fecha 2024-04-28 --hora 15
    python src/predecir.py --estacion "portal norte" --fecha 2024-05-01 --hora 18 --festivo
    python src/predecir.py --estacion ricaurte --fecha 2024-04-24 --dia-completo
    python src/predecir.py --listar
"""
from __future__ import annotations

import argparse
import difflib
import sys
from datetime import date

import joblib
import pandas as pd

from comun import (ALIAS_ESTACIONES, ANIO, CLASES, HORA_FIN, HORA_INICIO, MES, NOMBRES_DIA,
                   RUTA_MODELO, RUTA_PROCESADO, TIPOS_DIA, ULTIMO_DIA_ENTRENAMIENTO, construir_X, etiquetar, normalizar,
                   tipo_dia)


class ErrorEntrada(ValueError):
    """Error en los datos que escribió el usuario."""


def cargar_modelo(ruta=RUTA_MODELO) -> dict:
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecute primero: python src/entrenar_modelo.py")
    return joblib.load(ruta)


def buscar_estacion(paquete: dict, consulta: str) -> str:
    """Devuelve el código de la estación. Acepta el código, el nombre exacto o aproximado."""
    estaciones = paquete["estaciones"]
    consulta = consulta.strip()
    if consulta in estaciones:
        return consulta
    q = normalizar(consulta)
    if not q:
        raise ErrorEntrada("Debe indicar el nombre de una estación.")
    q = normalizar(ALIAS_ESTACIONES.get(q, q))
    nombres = {codigo: normalizar(info["estacion"]) for codigo, info in estaciones.items()}

    exactas = [c for c, n in nombres.items() if n == q]
    if len(exactas) == 1:
        return exactas[0]
    contienen = [c for c, n in nombres.items() if q in n]
    if len(contienen) == 1:
        return contienen[0]
    if len(contienen) > 1:
        opciones = ", ".join(sorted(estaciones[c]["estacion"] for c in contienen))
        raise ErrorEntrada(f"'{consulta}' es ambiguo. Coincide con: {opciones}")
    parecidas = difflib.get_close_matches(q, list(nombres.values()), n=1, cutoff=0.75)
    if parecidas:
        return next(c for c, n in nombres.items() if n == parecidas[0])
    sugerencias = difflib.get_close_matches(q, list(nombres.values()), n=3, cutoff=0.4)
    extra = ""
    if sugerencias:
        reales = [estaciones[c]["estacion"] for c, n in nombres.items() if n in sugerencias]
        extra = " ¿Quiso decir: " + ", ".join(reales) + "?"
    raise ErrorEntrada(f"No se encontró la estación '{consulta}'.{extra}")


def validar_hora(hora: int) -> int:
    if not HORA_INICIO <= hora <= HORA_FIN:
        raise ErrorEntrada(f"La hora debe estar entre {HORA_INICIO} y {HORA_FIN} "
                           f"(horario modelado de operación troncal).")
    return hora


def leer_fecha(texto: str) -> date:
    try:
        return date.fromisoformat(texto)
    except ValueError:
        raise ErrorEntrada(f"Fecha inválida '{texto}'. Use el formato AAAA-MM-DD.") from None


def predecir(paquete: dict, codigo: str, fecha: date, horas, festivo: bool = False) -> pd.DataFrame:
    info = paquete["estaciones"][codigo]
    t = 2 if festivo else tipo_dia(fecha)
    filas = pd.DataFrame({
        "hora": [validar_hora(h) for h in horas],
        # Abril de 2024 no tuvo festivos, así que un festivo se modela como un domingo
        "dia_semana": 6 if festivo else fecha.weekday(),
        "tipo_dia": t,
        "es_portal": info["es_portal"],
        "num_accesos": info["num_accesos"],
        "codigo_estacion": codigo,
        "zona": info["zona"],
    })
    X = construir_X(filas, paquete["demanda_media"], paquete["zonas"])[paquete["columnas"]]
    modelo = paquete["modelo"]
    prob = pd.DataFrame(modelo.predict_proba(X), columns=modelo.classes_)[CLASES]
    resultado = filas[["hora"]].copy()
    resultado["nivel"] = modelo.predict(X)
    resultado = pd.concat([resultado, prob.add_prefix("p_")], axis=1)
    resultado["regla_explicativa"] = [explicar(paquete, X.iloc[[i]]) for i in range(len(X))]
    return resultado


def explicar(paquete: dict, x: pd.DataFrame) -> str:
    """Camino que sigue el ejemplo en el árbol explicativo (profundidad 4)."""
    arbol = paquete["explicativo"]
    columnas = paquete["columnas"]
    t = arbol.tree_
    nodo, limites = 0, {}          # variable -> [límite inferior, límite superior]
    while t.children_left[nodo] != -1:
        var = columnas[t.feature[nodo]]
        umbral = float(t.threshold[nodo])
        bajo, alto = limites.get(var, [None, None])
        if float(x.iloc[0][var]) <= umbral:
            alto = umbral if alto is None else min(alto, umbral)
            nodo = t.children_left[nodo]
        else:
            bajo = umbral if bajo is None else max(bajo, umbral)
            nodo = t.children_right[nodo]
        limites[var] = [bajo, alto]
    partes = []
    for var, (bajo, alto) in limites.items():
        if bajo is not None and alto is not None:
            partes.append(f"{bajo:.1f} < {var} <= {alto:.1f}")
        elif alto is not None:
            partes.append(f"{var} <= {alto:.1f}")
        else:
            partes.append(f"{var} > {bajo:.1f}")
    clase = arbol.classes_[t.value[nodo][0].argmax()]
    return "SI " + " Y ".join(partes) + f" ENTONCES {clase}"


def valor_real(paquete: dict, codigo: str, fecha: date, horas) -> pd.DataFrame | None:
    """Si la fecha es de abril de 2024, devuelve las salidas reales para comparar."""
    if (fecha.year, fecha.month) != (ANIO, MES) or not RUTA_PROCESADO.exists():
        return None
    df = pd.read_csv(RUTA_PROCESADO, dtype={"codigo_estacion": str},
                     usecols=["codigo_estacion", "fecha", "hora", "salidas", "dia"])
    df = df[(df["codigo_estacion"] == codigo) & (df["fecha"] == fecha.isoformat())
            & (df["hora"].isin(list(horas)))].copy()
    if df.empty:
        return None
    df["nivel_real"] = etiquetar(df["salidas"], paquete["umbrales"])
    df["periodo"] = df["dia"].map(lambda d: "entrenamiento" if d <= ULTIMO_DIA_ENTRENAMIENTO else "prueba")
    return df.set_index("hora")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Predice el nivel de congestión de una estación "
                                            "troncal de TransMilenio.")
    p.add_argument("--estacion", help="Nombre o código de la estación (acepta nombres aproximados)")
    p.add_argument("--fecha", help="Fecha AAAA-MM-DD")
    p.add_argument("--hora", type=int, help=f"Hora entera entre {HORA_INICIO} y {HORA_FIN}")
    p.add_argument("--dia-completo", action="store_true", help="Predice todas las horas del día")
    p.add_argument("--festivo", action="store_true", help="Trata la fecha como festivo")
    p.add_argument("--listar", action="store_true", help="Lista las estaciones disponibles")
    args = p.parse_args(argv)

    try:
        paquete = cargar_modelo()
        if args.listar:
            for codigo, info in sorted(paquete["estaciones"].items(),
                                       key=lambda kv: (kv[1]["zona"], kv[1]["estacion"])):
                print(f"{codigo}  {info['estacion']:<45} {info['zona']}")
            return 0
        if not args.estacion or not args.fecha or (args.hora is None and not args.dia_completo):
            raise ErrorEntrada("Indique --estacion, --fecha y --hora (o --dia-completo).")

        codigo = buscar_estacion(paquete, args.estacion)
        fecha = leer_fecha(args.fecha)
        horas = range(HORA_INICIO, HORA_FIN + 1) if args.dia_completo else [args.hora]
        res = predecir(paquete, codigo, fecha, horas, args.festivo)
    except (ErrorEntrada, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    info = paquete["estaciones"][codigo]
    t = 2 if args.festivo else tipo_dia(fecha)
    q1, q2 = paquete["umbrales"]
    print(f"Estación: {info['estacion']} ({codigo}) – {info['zona']}")
    print(f"Fecha:    {fecha.isoformat()} ({NOMBRES_DIA[fecha.weekday()]}, {TIPOS_DIA[t]})")
    print(f"Niveles:  bajo <= {q1:.0f} < medio <= {q2:.0f} < alto  (salidas por hora)\n")
    real = valor_real(paquete, codigo, fecha, horas)
    if args.dia_completo:
        cab = "Hora  Nivel   P(bajo) P(medio) P(alto)"
        print(cab + ("   Salidas reales  Nivel real" if real is not None else ""))
        aciertos = 0
        for _, r in res.iterrows():
            h = int(r["hora"])
            barra = {"bajo": "▁  ", "medio": "▄▄ ", "alto": "███"}[r["nivel"]]
            linea = (f"{h:02d}:00 {r['nivel']:<7} {r['p_bajo']:>6.0%} {r['p_medio']:>8.0%} "
                     f"{r['p_alto']:>7.0%}  {barra}")
            if real is not None and h in real.index:
                ok = real.at[h, "nivel_real"] == r["nivel"]
                aciertos += ok
                linea += f" {real.at[h, 'salidas']:>8,}  {real.at[h, 'nivel_real']:<6} {'✓' if ok else '✗'}"
            print(linea)
        if real is not None:
            print(f"\nAciertos: {aciertos}/{len(res)} horas "
                  f"(día del periodo de {real['periodo'].iloc[0]})")
    else:
        r = res.iloc[0]
        h = int(r["hora"])
        print(f"Hora {h:02d}:00 -> congestión {r['nivel'].upper()}")
        print(f"Probabilidades: bajo {r['p_bajo']:.0%} | medio {r['p_medio']:.0%} | "
              f"alto {r['p_alto']:.0%}")
        print(f"Explicación (árbol de profundidad 4): {r['regla_explicativa']}")
        if real is not None and h in real.index:
            ok = real.at[h, "nivel_real"] == r["nivel"]
            print(f"Valor real: {real.at[h, 'salidas']:,} salidas -> {real.at[h, 'nivel_real']} "
                  f"({'acierto' if ok else 'error'}; día del periodo de {real.at[h, 'periodo']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
