from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL

DATA_DIR = Path("dados_olist")
OUTPUT_DIR = Path("dados_olist")


def carregar_dados_classificacao(data_dir: Path) -> pd.DataFrame:
    orders = pd.read_csv(
        data_dir / "olist_orders_dataset.csv",
        parse_dates=[
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    reviews = pd.read_csv(data_dir / "olist_order_reviews_dataset.csv")
    items = pd.read_csv(data_dir / "olist_order_items_dataset.csv")

    # Filtra apenas pedidos entregues com datas validas
    orders = orders[orders["order_status"] == "delivered"].copy()
    orders = orders.dropna(
        subset=[
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
    )

    # Agrega itens por pedido
    items_agg = (
        items.groupby("order_id", as_index=False)
        .agg(
            price=("price", "sum"),
            freight_value=("freight_value", "sum"),
            qtd_itens=("order_item_id", "count"),
        )
    )

    # Mantem uma avaliacao por pedido
    reviews_agg = (
        reviews.dropna(subset=["review_score"])
        .groupby("order_id", as_index=False)["review_score"]
        .mean()
    )

    df = orders.merge(items_agg, on="order_id", how="inner").merge(
        reviews_agg, on="order_id", how="inner"
    )

    # Engenharia de atributos
    df["tempo_transporte_dias"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0
    df["atraso_dias"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400.0
    df["is_late"] = (df["atraso_dias"] > 0).astype(int)
    df["freight_ratio"] = np.where(df["price"] > 0, df["freight_value"] / df["price"], 0.0)
    df["is_detractor"] = (df["review_score"] <= 2).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "price",
            "freight_value",
            "qtd_itens",
            "tempo_transporte_dias",
            "atraso_dias",
            "freight_ratio",
            "is_detractor",
        ]
    )
    return df


def escolher_melhor_threshold(y_val: pd.Series, probs_val: np.ndarray) -> float:
    melhor_t = 0.5
    melhor_f1 = -1.0
    for t in np.arange(0.10, 0.85, 0.01):
        preds = (probs_val >= t).astype(int)
        score = f1_score(y_val, preds, zero_division=0)
        if score > melhor_f1:
            melhor_f1 = score
            melhor_t = float(t)
    return melhor_t


def avaliar_classificacao(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    df = carregar_dados_classificacao(data_dir)
    features = [
        "price",
        "freight_value",
        "qtd_itens",
        "tempo_transporte_dias",
        "atraso_dias",
        "is_late",
        "freight_ratio",
    ]
    X = df[features]
    y = df["is_detractor"]

    # Divisao sem data leakage: 70% treino, 15% validacao e 15% teste
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    resultados = []

    # 1. Baseline Ingenuo (classe majoritaria)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    resultados.append(
        {
            "modelo": "Baseline Ingenuo (Maior Incidencia)",
            "acuracia": round(accuracy_score(y_test, y_pred_dummy), 4),
            "precisao_detrator": round(precision_score(y_test, y_pred_dummy, zero_division=0), 4),
            "recall_detrator": round(recall_score(y_test, y_pred_dummy, zero_division=0), 4),
            "f1_score_detrator": round(f1_score(y_test, y_pred_dummy, zero_division=0), 4),
            "roc_auc": 0.5000,
            "threshold_validacao": 0.50,
            "pedidos_validos": len(df),
        }
    )

    # 2. Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    probs_val_rf = rf_model.predict_proba(X_val)[:, 1]
    t_rf = escolher_melhor_threshold(y_val, probs_val_rf)
    probs_test_rf = rf_model.predict_proba(X_test)[:, 1]
    y_pred_rf = (probs_test_rf >= t_rf).astype(int)
    resultados.append(
        {
            "modelo": "Random Forest Classifier",
            "acuracia": round(accuracy_score(y_test, y_pred_rf), 4),
            "precisao_detrator": round(precision_score(y_test, y_pred_rf, zero_division=0), 4),
            "recall_detrator": round(recall_score(y_test, y_pred_rf, zero_division=0), 4),
            "f1_score_detrator": round(f1_score(y_test, y_pred_rf, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, probs_test_rf), 4),
            "threshold_validacao": round(t_rf, 2),
            "pedidos_validos": len(df),
        }
    )

    # 3. Regressao Logistica (RobustScaler ajustado apenas no treino)
    lr_pipe = Pipeline(
        [
            ("scaler", RobustScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    lr_pipe.fit(X_train, y_train)
    probs_val_lr = lr_pipe.predict_proba(X_val)[:, 1]
    t_lr = escolher_melhor_threshold(y_val, probs_val_lr)
    probs_test_lr = lr_pipe.predict_proba(X_test)[:, 1]
    y_pred_lr = (probs_test_lr >= t_lr).astype(int)
    resultados.append(
        {
            "modelo": "Regressao Logistica",
            "acuracia": round(accuracy_score(y_test, y_pred_lr), 4),
            "precisao_detrator": round(precision_score(y_test, y_pred_lr, zero_division=0), 4),
            "recall_detrator": round(recall_score(y_test, y_pred_lr, zero_division=0), 4),
            "f1_score_detrator": round(f1_score(y_test, y_pred_lr, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, probs_test_lr), 4),
            "threshold_validacao": round(t_lr, 2),
            "pedidos_validos": len(df),
        }
    )

    df_res = pd.DataFrame(resultados)
    output_dir.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(output_dir / "metricas_classificacao_olist.csv", index=False)
    print("Metricas de classificacao salvas:")
    print(df_res.to_string(index=False))
    return df_res


def calcular_mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    mask = y_true > 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def calcular_wape(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float((np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))) * 100.0)


def avaliar_series_temporais(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    orders = pd.read_csv(
        data_dir / "olist_orders_dataset.csv",
        parse_dates=["order_purchase_timestamp"],
    )
    items = pd.read_csv(data_dir / "olist_order_items_dataset.csv")

    orders = orders[orders["order_status"] != "canceled"].copy()
    merged = orders[["order_id", "order_purchase_timestamp"]].merge(
        items[["order_id", "price"]], on="order_id", how="inner"
    )
    merged["data"] = merged["order_purchase_timestamp"].dt.floor("D")

    serie_diaria = (
        merged.groupby("data")["price"]
        .sum()
        .asfreq("D", fill_value=0.0)
        .loc["2017-01-01":"2018-08-31"]
    )

    # Decomposicao STL da serie diaria
    _ = STL(serie_diaria, period=7, robust=True).fit()

    # Holdout Temporal: ultimos 6 meses (marco a agosto de 2018) para teste
    train_ts = serie_diaria.loc[:"2018-02-28"]
    test_ts = serie_diaria.loc["2018-03-01":"2018-08-31"]

    # 1. SARIMAX
    sarimax_model = sm.tsa.statespace.SARIMAX(
        train_ts,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    sarimax_fit = sarimax_model.fit(disp=False)
    pred_sarimax = sarimax_fit.predict(
        start=len(train_ts), end=len(train_ts) + len(test_ts) - 1
    )
    pred_sarimax.index = test_ts.index

    # 2. Referencia: Media historica por dia da semana
    media_dow = train_ts.groupby(train_ts.index.dayofweek).mean()
    pred_dow = pd.Series(test_ts.index.dayofweek.map(media_dow), index=test_ts.index)

    # 3. Referencia: Media das ultimas 4 semanas do treino
    ultimas_4_sem = train_ts.iloc[-28:]
    media_4_sem = ultimas_4_sem.groupby(ultimas_4_sem.index.dayofweek).mean()
    pred_4_sem = pd.Series(test_ts.index.dayofweek.map(media_4_sem), index=test_ts.index)

    comparativos = [
        ("SARIMAX", pred_sarimax),
        ("Media historica por dia da semana", pred_dow),
        ("Media das ultimas quatro semanas do treino", pred_4_sem),
    ]

    linhas = []
    for nome, pred in comparativos:
        linhas.append(
            {
                "metodo_previsao": nome,
                "mape_teste_pct": round(calcular_mape(test_ts, pred), 2),
                "wape_teste_pct": round(calcular_wape(test_ts, pred), 2),
                "r2_teste": round(float(r2_score(test_ts, pred)), 4),
            }
        )

    df_ts = pd.DataFrame(linhas)
    output_dir.mkdir(parents=True, exist_ok=True)
    df_ts.to_csv(output_dir / "metricas_sazonalidade_olist.csv", index=False)
    print("\nMetricas de series temporais salvas:")
    print(df_ts.to_string(index=False))
    return df_ts


if __name__ == "__main__":
    avaliar_classificacao(DATA_DIR, OUTPUT_DIR)
    avaliar_series_temporais(DATA_DIR, OUTPUT_DIR)
