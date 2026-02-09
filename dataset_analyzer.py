import pandas as pd


def analizar_dataframe(df, tipo_tarea):
    filas, columnas = df.shape

    ratio_missing = df.isnull().sum().sum() / (filas * columnas) if filas > 0 else 0

    cols_numericas = df.select_dtypes(include=["number"]).shape[1]
    cols_categoricas = df.select_dtypes(include=["object", "category"]).shape[1]
    ratio_numericas = cols_numericas / columnas if columnas > 0 else 0
    ratio_categoricas = cols_categoricas / columnas if columnas > 0 else 0

    dimensionalidad_ratio = columnas / filas if filas > 0 else 0

    cardinalidad_promedio = 0
    if cols_categoricas > 0:
        cardinalidades = [
            df[col].nunique()
            for col in df.select_dtypes(include=["object", "category"]).columns
        ]
        cardinalidad_promedio = (
            sum(cardinalidades) / len(cardinalidades) if cardinalidades else 0
        )

    metrics = {
        "filas": filas,
        "cols": columnas,
        "ratio_missing": ratio_missing,
        "cols_numericas": cols_numericas,
        "cols_categoricas": cols_categoricas,
        "ratio_numericas": ratio_numericas,
        "ratio_categoricas": ratio_categoricas,
        "dimensionalidad_ratio": dimensionalidad_ratio,
        "cardinalidad_promedio": cardinalidad_promedio,
        "info_extra": None,
        "razon_dificultad": [],
    }

    if tipo_tarea == "classification":
        posibles_targets = [c for c in df.columns if 2 <= df[c].nunique() <= 20]

        if posibles_targets:
            target = min(posibles_targets, key=lambda c: df[c].nunique())
            balance = df[target].value_counts(normalize=True).max()
            n_clases = df[target].nunique()
            metrics["desbalance"] = balance
            metrics["n_clases"] = n_clases
            metrics["info_extra"] = f"classes: {n_clases}, balance: {balance:.2f}"
        else:
            metrics["desbalance"] = 0
            metrics["n_clases"] = 0
            metrics["info_extra"] = "target no detectado"

    elif tipo_tarea == "regression":
        posibles_targets_num = df.select_dtypes(include=["number"]).columns
        if len(posibles_targets_num) > 0:
            target_col = posibles_targets_num[-1]
            target_std = df[target_col].std()
            target_range = df[target_col].max() - df[target_col].min()
            metrics["target_std"] = target_std
            metrics["target_range"] = target_range
            metrics["info_extra"] = (
                f"num: {cols_numericas}/{columnas}, std: {target_std:.2f}"
            )
        else:
            metrics["info_extra"] = f"num: {cols_numericas}/{columnas}"

    return metrics


def determinar_nivel(metrics, tipo_tarea):
    if not metrics:
        return "error", ["error al leer dataset"]

    razones = []
    puntos_dificultad = 0

    if metrics["ratio_missing"] > 0.15:
        puntos_dificultad += 3
        razones.append(f"alto missing data ({metrics['ratio_missing']:.1%})")
    elif metrics["ratio_missing"] > 0.05:
        puntos_dificultad += 1
        razones.append(f"moderado missing data ({metrics['ratio_missing']:.1%})")

    if metrics["ratio_numericas"] < 0.3:
        puntos_dificultad += 3
        razones.append(f"pocas features numericas ({metrics['ratio_numericas']:.1%})")
    elif metrics["ratio_numericas"] < 0.7:
        puntos_dificultad += 1
        razones.append("mix de tipos de datos")

    if metrics["cols"] > 100:
        puntos_dificultad += 3
        razones.append(f"alta dimensionalidad ({metrics['cols']} features)")
    elif metrics["cols"] > 50:
        puntos_dificultad += 1
        razones.append(f"dimensionalidad media ({metrics['cols']} features)")

    if metrics["dimensionalidad_ratio"] > 0.1:
        puntos_dificultad += 2
        razones.append(
            f"ratio features/samples alto ({metrics['dimensionalidad_ratio']:.3f})"
        )

    if metrics["cardinalidad_promedio"] > 50:
        puntos_dificultad += 2
        razones.append(
            f"alta cardinalidad categorica (avg: {metrics['cardinalidad_promedio']:.0f})"
        )

    if tipo_tarea == "classification":
        desbalance = metrics.get("desbalance", 0)
        if desbalance > 0.85:
            puntos_dificultad += 3
            razones.append(f"clases muy desbalanceadas ({desbalance:.2f})")
        elif desbalance > 0.65:
            puntos_dificultad += 2
            razones.append(f"clases moderadamente desbalanceadas ({desbalance:.2f})")

        n_clases = metrics.get("n_clases", 0)
        if n_clases > 10:
            puntos_dificultad += 2
            razones.append(f"multiclase ({n_clases} clases)")
        elif n_clases > 5:
            puntos_dificultad += 1
            razones.append(f"multiclase ({n_clases} clases)")

    elif tipo_tarea == "regression" and "ratio_numericas" in metrics:
        if metrics["ratio_numericas"] < 0.5:
            puntos_dificultad += 2
            razones.append("pocas variables numericas para regresion")

    # easy: 0-2, mid: 3-5, hard: 6+
    if puntos_dificultad <= 2:
        nivel = "easy"
        if not razones:
            razones.append("dataset limpio y bien estructurado")
    elif puntos_dificultad <= 5:
        nivel = "mid"
    else:
        nivel = "hard"

    metrics["razon_dificultad"] = razones
    metrics["puntos_dificultad"] = puntos_dificultad

    return nivel, razones


def print_metrics(metricas, nivel, razones):
    """print dataset analysis results"""
    print(f"   dimensions: {metricas['filas']} rows x {metricas['cols']} columns")
    print(
        f"   features: {metricas['cols_numericas']} numeric, {metricas['cols_categoricas']} categorical"
    )
    print(f"   level: {nivel} (score: {metricas['puntos_dificultad']})")
    print("   reasons:")
    for razon in razones:
        print(f"      - {razon}")
