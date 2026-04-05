import pandas as pd
import numpy as np
from config import settings


def analizar_dataframe(df, tipo_tarea):
    filas, columnas = df.shape

    ratio_missing = df.isnull().sum().sum() / (filas * columnas) if filas > 0 else 0
    ratio_invalid = _calcular_ratio_datos_invalidos(df)

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
        "ratio_invalid": ratio_invalid,
        "cols_numericas": cols_numericas,
        "cols_categoricas": cols_categoricas,
        "ratio_numericas": ratio_numericas,
        "ratio_categoricas": ratio_categoricas,
        "dimensionalidad_ratio": dimensionalidad_ratio,
        "cardinalidad_promedio": cardinalidad_promedio,
        "info_extra": None,
        "razon_dificultad": [],
        "target_detected": False,
    }

    if tipo_tarea == "classification":
        target = _infer_target_col(df, tipo_tarea)

        if target:
            balance = df[target].value_counts(normalize=True).max()
            n_clases = df[target].nunique()
            metrics["desbalance"] = balance
            metrics["n_clases"] = n_clases
            metrics["target_detected"] = True
            metrics["info_extra"] = f"classes: {n_clases}, balance: {balance:.2f}"
        else:
            metrics["desbalance"] = 0
            metrics["n_clases"] = 0
            metrics["info_extra"] = "target no detectado"

    elif tipo_tarea == "regression":
        target_col = _infer_target_col(df, tipo_tarea)
        if target_col:
            target_std = df[target_col].std()
            target_range = df[target_col].max() - df[target_col].min()
            metrics["target_std"] = target_std
            metrics["target_range"] = target_range
            metrics["target_detected"] = True
            metrics["info_extra"] = (
                f"num: {cols_numericas}/{columnas}, std: {target_std:.2f}"
            )
        else:
            metrics["info_extra"] = f"num: {cols_numericas}/{columnas}"

    return metrics


def determinar_nivel(metrics, tipo_tarea):
    if not metrics:
        return "error", ["error al leer dataset"]

    if _es_easy_estricto(metrics, tipo_tarea):
        razones = [
            "all features are numeric",
            f"very few columns ({metrics['cols']})",
            "0% missing data",
            "0% invalid values",
        ]
        if tipo_tarea == "classification":
            razones.append(f"low class imbalance ({metrics.get('desbalance', 0):.2f})")

        metrics["razon_dificultad"] = razones
        metrics["puntos_dificultad"] = 0
        return "easy", razones

    razones = []
    puntos_dificultad = 0
    cfg = settings.analyzer

    if metrics["ratio_missing"] > cfg.high_missing_ratio:
        puntos_dificultad += 3
        razones.append(f"alto missing data ({metrics['ratio_missing']:.1%})")
    elif metrics["ratio_missing"] > cfg.moderate_missing_ratio:
        puntos_dificultad += 1
        razones.append(f"moderado missing data ({metrics['ratio_missing']:.1%})")

    if metrics["ratio_invalid"] > cfg.high_invalid_ratio:
        puntos_dificultad += 3
        razones.append(f"muchos valores invalidos ({metrics['ratio_invalid']:.1%})")
    elif metrics["ratio_invalid"] > cfg.moderate_invalid_ratio:
        puntos_dificultad += 1
        razones.append(f"algunos valores invalidos ({metrics['ratio_invalid']:.1%})")

    if metrics["ratio_numericas"] < cfg.low_numeric_ratio:
        puntos_dificultad += 3
        razones.append(f"pocas features numericas ({metrics['ratio_numericas']:.1%})")
    elif metrics["ratio_numericas"] < cfg.medium_numeric_ratio:
        puntos_dificultad += 1
        razones.append("mix de tipos de datos")

    if metrics["cols"] > cfg.high_columns:
        puntos_dificultad += 3
        razones.append(f"alta dimensionalidad ({metrics['cols']} features)")
    elif metrics["cols"] > cfg.medium_columns:
        puntos_dificultad += 1
        razones.append(f"dimensionalidad media ({metrics['cols']} features)")

    if metrics["dimensionalidad_ratio"] > cfg.high_feature_sample_ratio:
        puntos_dificultad += 2
        razones.append(
            f"ratio features/samples muy alto ({metrics['dimensionalidad_ratio']:.3f})"
        )
    elif metrics["dimensionalidad_ratio"] > cfg.medium_feature_sample_ratio:
        puntos_dificultad += 1
        razones.append(
            f"ratio features/samples alto ({metrics['dimensionalidad_ratio']:.3f})"
        )

    if metrics["cardinalidad_promedio"] > cfg.high_categorical_cardinality:
        puntos_dificultad += 2
        razones.append(
            f"alta cardinalidad categorica (avg: {metrics['cardinalidad_promedio']:.0f})"
        )
    elif metrics["cardinalidad_promedio"] > cfg.medium_categorical_cardinality:
        puntos_dificultad += 1
        razones.append(
            f"cardinalidad categorica media (avg: {metrics['cardinalidad_promedio']:.0f})"
        )

    if tipo_tarea == "classification":
        if not metrics.get("target_detected", False):
            puntos_dificultad += 2
            razones.append("target no detectado con alta confianza")

        desbalance = metrics.get("desbalance", 0)
        if desbalance > cfg.severe_imbalance:
            puntos_dificultad += 3
            razones.append(f"clases muy desbalanceadas ({desbalance:.2f})")
        elif desbalance > cfg.high_imbalance:
            puntos_dificultad += 2
            razones.append(f"clases moderadamente desbalanceadas ({desbalance:.2f})")
        elif desbalance > cfg.moderate_imbalance:
            puntos_dificultad += 1
            razones.append(f"ligero desbalance de clases ({desbalance:.2f})")

        n_clases = metrics.get("n_clases", 0)
        if n_clases > cfg.high_multiclass:
            puntos_dificultad += 2
            razones.append(f"multiclase ({n_clases} clases)")
        elif n_clases > cfg.medium_multiclass:
            puntos_dificultad += 1
            razones.append(f"multiclase ({n_clases} clases)")

    elif tipo_tarea == "regression" and "ratio_numericas" in metrics:
        if not metrics.get("target_detected", False):
            puntos_dificultad += 1
            razones.append("target no detectado con alta confianza")

        if metrics["ratio_numericas"] < cfg.low_numeric_ratio:
            puntos_dificultad += 2
            razones.append("pocas variables numericas para regresion")

    if not razones:
        razones.append("no cumple criterios estrictos de easy")

    nivel = "hard" if puntos_dificultad >= cfg.hard_min_points else "mid"

    metrics["razon_dificultad"] = razones
    metrics["puntos_dificultad"] = puntos_dificultad

    return nivel, razones


def _infer_target_col(df, tipo_tarea):
    if tipo_tarea == "classification":
        hints = ("target", "label", "class", "y", "outcome")
        for col in df.columns:
            if _matches_hint(col, hints):
                nunique = df[col].nunique(dropna=True)
                if 2 <= nunique <= 100:
                    return col

        posibles_targets = [
            c for c in df.columns if 2 <= df[c].nunique(dropna=True) <= 30
        ]
        if posibles_targets:
            return min(posibles_targets, key=lambda c: df[c].nunique(dropna=True))
        return None

    if tipo_tarea == "regression":
        hints = ("target", "y", "price", "salary", "score")
        numeric_cols = list(df.select_dtypes(include=["number"]).columns)
        for col in numeric_cols:
            if _matches_hint(col, hints):
                return col
        return numeric_cols[-1] if numeric_cols else None

    return None


def _calcular_ratio_datos_invalidos(df):
    filas, columnas = df.shape
    total_celdas = filas * columnas
    if total_celdas == 0:
        return 0.0

    invalidos = 0

    num_df = df.select_dtypes(include=["number"])
    if not num_df.empty:
        num_values = num_df.to_numpy(dtype=float, copy=False)
        invalid_num = (~pd.isna(num_values)) & (~np.isfinite(num_values))
        invalidos += int(invalid_num.sum())

    obj_df = df.select_dtypes(include=["object", "category"])
    if not obj_df.empty:
        tokens_invalidos = {"", "na", "n/a", "null", "none", "nan", "?", "-"}
        raw_obj = obj_df.astype("string")
        cleaned = raw_obj.apply(lambda s: s.str.strip().str.lower())
        invalid_obj = raw_obj.notna() & cleaned.isin(tokens_invalidos)
        invalidos += int(invalid_obj.sum().sum())

    return invalidos / total_celdas


def _es_easy_estricto(metrics, tipo_tarea):
    cfg = settings.analyzer
    numeric_only = (
        metrics["ratio_numericas"] == 1.0 if cfg.easy_require_numeric_only else True
    )
    base_ok = (
        numeric_only
        and metrics["cols"] <= cfg.easy_max_columns
        and metrics["ratio_missing"] <= cfg.easy_required_missing_ratio
        and metrics["ratio_invalid"] <= cfg.easy_required_invalid_ratio
    )
    if not base_ok:
        return False

    if tipo_tarea == "classification":
        return (
            metrics.get("target_detected", False)
            and metrics.get("desbalance", 1.0) <= cfg.easy_max_class_majority
        )

    return True


def _matches_hint(column_name, hints):
    col = str(column_name).strip().lower().replace(" ", "_")
    if col in hints:
        return True

    for hint in hints:
        if len(hint) <= 1:
            continue
        if col.startswith(f"{hint}_") or col.endswith(f"_{hint}"):
            return True

    return False


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
