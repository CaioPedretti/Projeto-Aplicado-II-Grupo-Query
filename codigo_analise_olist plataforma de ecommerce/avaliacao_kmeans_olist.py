from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("dados_olist")
OUTPUT_DIR = Path("dados_olist")


def distancia_haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


def carregar_dados_geograficos(data_dir: Path) -> pd.DataFrame:
    orders = pd.read_csv(data_dir / "olist_orders_dataset.csv")
    items = pd.read_csv(data_dir / "olist_order_items_dataset.csv")
    customers = pd.read_csv(data_dir / "olist_customers_dataset.csv")
    sellers = pd.read_csv(data_dir / "olist_sellers_dataset.csv")
    geo = pd.read_csv(data_dir / "olist_geolocation_dataset.csv")

    # Coordenada media por prefixo de CEP
    geo_agg = (
        geo.groupby("geolocation_zip_code_prefix", as_index=False)
        .agg(lat=("geolocation_lat", "mean"), lng=("geolocation_lng", "mean"))
    )

    orders_delivered = orders[orders["order_status"] == "delivered"][
        ["order_id", "customer_id"]
    ]

    df = (
        items[["order_id", "seller_id", "price", "freight_value"]]
        .merge(orders_delivered, on="order_id", how="inner")
        .merge(
            customers[["customer_id", "customer_zip_code_prefix", "customer_state"]],
            on="customer_id",
            how="inner",
        )
        .merge(
            sellers[["seller_id", "seller_zip_code_prefix", "seller_state"]],
            on="seller_id",
            how="inner",
        )
        .merge(
            geo_agg.rename(
                columns={
                    "geolocation_zip_code_prefix": "customer_zip_code_prefix",
                    "lat": "cust_lat",
                    "lng": "cust_lng",
                }
            ),
            on="customer_zip_code_prefix",
            how="inner",
        )
        .merge(
            geo_agg.rename(
                columns={
                    "geolocation_zip_code_prefix": "seller_zip_code_prefix",
                    "lat": "sell_lat",
                    "lng": "sell_lng",
                }
            ),
            on="seller_zip_code_prefix",
            how="inner",
        )
    )

    df = df[df["price"] > 0].copy()
    df["distancia_km"] = distancia_haversine(
        df["sell_lat"], df["sell_lng"], df["cust_lat"], df["cust_lng"]
    )
    df["freight_ratio"] = df["freight_value"] / df["price"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["distancia_km", "freight_value", "freight_ratio"]
    )
    return df


def executar_kmeans(data_dir: Path, output_dir: Path) -> None:
    df = carregar_dados_geograficos(data_dir)
    features = ["distancia_km", "freight_value", "freight_ratio"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    # Avaliacao de k = 3, 4 e 5 com silhueta em amostra fixa de 2.500 itens
    metricas = []
    clusters_k4 = None

    for k in [3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(
            X_scaled, labels, sample_size=min(2500, len(df)), random_state=42
        )
        metricas.append(
            {
                "num_clusters_k": k,
                "silhueta_amostral": round(float(sil), 4),
                "inercia_wcss": round(float(km.inertia_), 2),
                "itens_validos": len(df),
            }
        )
        if k == 4:
            clusters_k4 = labels

    df["cluster"] = clusters_k4
    resumo = (
        df.groupby("cluster", as_index=False)
        .agg(
            quantidade_itens=("order_id", "count"),
            distancia_media_km=("distancia_km", "mean"),
            frete_medio_brl=("freight_value", "mean"),
            razao_frete_preco_pct=("freight_ratio", lambda s: s.mean() * 100.0),
        )
    )
    resumo["distancia_media_km"] = resumo["distancia_media_km"].round(0).astype(int)
    resumo["frete_medio_brl"] = resumo["frete_medio_brl"].round(2)
    resumo["razao_frete_preco_pct"] = resumo["razao_frete_preco_pct"].round(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    df_metricas = pd.DataFrame(metricas)
    df_metricas.to_csv(output_dir / "metricas_kmeans_olist.csv", index=False)
    resumo.to_csv(output_dir / "resumo_clusters_olist.csv", index=False)

    print("Metricas K-Means salvas:")
    print(df_metricas.to_string(index=False))
    print("\nResumo dos clusters (k=4):")
    print(resumo.to_string(index=False))


if __name__ == "__main__":
    executar_kmeans(DATA_DIR, OUTPUT_DIR)
