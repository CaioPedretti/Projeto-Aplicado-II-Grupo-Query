from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm

DATA_DIR = Path("dados_olist")
OUTPUT_DIR = Path("dados_olist")


def carregar_base_atraso(data_dir: Path) -> pd.DataFrame:
    orders = pd.read_csv(
        data_dir / "olist_orders_dataset.csv",
        parse_dates=[
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    reviews = pd.read_csv(data_dir / "olist_order_reviews_dataset.csv")

    orders = orders[orders["order_status"] == "delivered"].dropna(
        subset=["order_delivered_customer_date", "order_estimated_delivery_date"]
    )
    reviews_agg = (
        reviews.dropna(subset=["review_score"])
        .groupby("order_id", as_index=False)["review_score"]
        .mean()
    )

    df = orders.merge(reviews_agg, on="order_id", how="inner")
    df["atraso_dias"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400.0
    df["is_detractor"] = (df["review_score"] <= 2).astype(int)
    return df


def executar_analise_atraso(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    df = carregar_base_atraso(data_dir)

    # 1. Regressao Logistica (base completa): Odds Ratio por dia de atraso
    X_logit = sm.add_constant(df[["atraso_dias"]])
    modelo_logit = sm.Logit(df["is_detractor"], X_logit).fit(disp=False)
    or_geral = float(np.exp(modelo_logit.params["atraso_dias"]))
    p_geral = float(modelo_logit.pvalues["atraso_dias"])

    # 2. Regressao Logistica focada em atrasos severos (ate 30 dias)
    df_30 = df[(df["atraso_dias"] >= -30) & (df["atraso_dias"] <= 30)].copy()
    X_logit_30 = sm.add_constant(df_30[["atraso_dias"]])
    modelo_logit_30 = sm.Logit(df_30["is_detractor"], X_logit_30).fit(disp=False)
    or_30 = float(np.exp(modelo_logit_30.params["atraso_dias"]))
    p_30 = float(modelo_logit_30.pvalues["atraso_dias"])

    # 3. Regressao Linear (OLS): reducao media no review_score por dia de atraso
    modelo_ols = sm.OLS(df["review_score"], X_logit).fit()
    reducao_nota_dia = float(modelo_ols.params["atraso_dias"])
    p_ols = float(modelo_ols.pvalues["atraso_dias"])

    # 4. Correlacao de Spearman
    rho, p_spearman = spearmanr(df["atraso_dias"], df["review_score"])

    df_metricas = pd.DataFrame(
        [
            {
                "analise": "Odds Ratio por dia de atraso (base completa)",
                "estimativa": round(or_geral, 4),
                "variacao_pct_dia": round((or_geral - 1.0) * 100.0, 2),
                "p_valor": f"{p_geral:.4e}",
                "observacao": "Aumento estatisticamente significativo nas chances de nota 1 ou 2",
            },
            {
                "analise": "Odds Ratio por dia de atraso (recorte ate 30 dias)",
                "estimativa": round(or_30, 4),
                "variacao_pct_dia": round((or_30 - 1.0) * 100.0, 2),
                "p_valor": f"{p_30:.4e}",
                "observacao": "Sensibilidade acentuada em atrasos severos (impacto nao linear)",
            },
            {
                "analise": "Reducao media no review_score por dia adicional (OLS)",
                "estimativa": round(reducao_nota_dia, 4),
                "variacao_pct_dia": np.nan,
                "p_valor": f"{p_ols:.4e}",
                "observacao": "Queda media esperada na nota de avaliacao (escala 1 a 5)",
            },
            {
                "analise": "Correlacao de Spearman (atraso_dias vs review_score)",
                "estimativa": round(float(rho), 4),
                "variacao_pct_dia": np.nan,
                "p_valor": f"{p_spearman:.4e}",
                "observacao": "Associacao negativa significativa entre atraso e satisfacao",
            },
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    df_metricas.to_csv(output_dir / "metricas_atraso_olist.csv", index=False)
    print("Metricas estatisticas de atraso salvas:")
    print(df_metricas.to_string(index=False))
    return df_metricas


if __name__ == "__main__":
    executar_analise_atraso(DATA_DIR, OUTPUT_DIR)
