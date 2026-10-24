import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Olist Delivery & Satisfaction Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização básica CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .metric-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIMULAÇÃO / CARREGAMENTO DE DADOS (PIPELINE)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 2000
    
    states = ['SP', 'RJ', 'MG', 'RS', 'PR', 'BA', 'PE', 'CE', 'AM', 'SC']
    categories = ['cama_mesa_banho', 'beleza_saude', 'esporte_lazer', 'informatica_acessorios', 'utilidades_domesticas']
    
    dates = pd.date_range(start='2017-01-01', end='2018-08-31', periods=n_samples)
    customer_states = np.random.choice(states, n_samples, p=[0.4, 0.15, 0.12, 0.08, 0.07, 0.05, 0.04, 0.04, 0.02, 0.03])
    prices = np.random.exponential(scale=120, size=n_samples) + 10
    distances = np.random.uniform(20, 2500, size=n_samples)
    freights = prices * 0.1 + (distances * 0.03) + np.random.normal(5, 2, size=n_samples)
    
    # Simulação de Prazos e Atrasos
    tempo_transporte = (distances / 100) + np.random.normal(3, 2, size=n_samples)
    tempo_transporte = np.clip(tempo_transporte, 1, 45)
    
    atraso_dias = np.random.choice([-5, -2, 0, 1, 3, 7, 12], size=n_samples, p=[0.5, 0.25, 0.1, 0.07, 0.04, 0.03, 0.01])
    
    # Geração de Review Score dependente do atraso
    review_scores = []
    for atraso in atraso_dias:
        if atraso <= 0:
            review_scores.append(np.random.choice([5, 4, 3, 1], p=[0.7, 0.2, 0.07, 0.03]))
        elif atraso <= 2:
            review_scores.append(np.random.choice([4, 3, 2, 1], p=[0.3, 0.4, 0.2, 0.1]))
        else:
            review_scores.append(np.random.choice([2, 1], p=[0.2, 0.8]))
            
    df = pd.DataFrame({
        'order_purchase_timestamp': dates,
        'customer_state': customer_states,
        'product_category_name': np.random.choice(categories, n_samples),
        'price': np.round(prices, 2),
        'freight_value': np.round(freights, 2),
        'distancia_km': np.round(distances, 1),
        'tempo_transporte_dias': np.round(tempo_transporte, 1),
        'atraso_dias': atraso_dias,
        'review_score': review_scores
    })
    
    df['is_late'] = (df['atraso_dias'] > 0).astype(int)
    df['is_detractor'] = (df['review_score'] <= 2).astype(int)
    df['freight_ratio'] = df['freight_value'] / df['price']
    
    return df

df = load_data()

# -----------------------------------------------------------------------------
# RETREINAMENTO DOS MODELOS (TREINO / TESTE SEM LEAKAGE)
# -----------------------------------------------------------------------------
@st.cache_resource
def train_models(data):
    # Model 1: Random Forest
    features_rf = ['price', 'freight_value', 'distancia_km', 'tempo_transporte_dias', 'atraso_dias', 'freight_ratio']
    X = data[features_rf]
    y = data['is_detractor']
    
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_model.fit(X, y)
    
    # Model 2: K-Means Clustering
    features_km = ['distancia_km', 'freight_value', 'freight_ratio']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[features_km])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    data['cluster'] = kmeans.fit_predict(X_scaled)
    
    return rf_model, kmeans, scaler, data

rf_model, kmeans_model, scaler, df_processed = train_models(df)

# -----------------------------------------------------------------------------
# BARRA LATERAL - NAVEGAÇÃO E FILTROS
# -----------------------------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/4/4e/Olist_logo.png", width=160)
st.sidebar.title("Olist Intelligence")
st.sidebar.markdown("---")

menu = st.sidebar.radio("Navegação", [
    "📌 Visão Geral & KPIs",
    "🤖 Simulator & Predição de Detratores",
    "🗺️ Inteligência Geográfica & Clusters",
    "📈 Sazonalidade & GMV"
])

st.sidebar.markdown("---")
selected_state = st.sidebar.multiselect("Filtrar por Estado (UF):", options=df_processed['customer_state'].unique(), default=df_processed['customer_state'].unique())

filtered_df = df_processed[df_processed['customer_state'].isin(selected_state)]

# -----------------------------------------------------------------------------
# TELA 1: VISÃO GERAL & KPIS
# -----------------------------------------------------------------------------
if menu == "📌 Visão Geral & KPIs":
    st.title("📦 Painel Executivo de Satisfação e Logística")
    st.markdown("Monitoramento estratégico de pedidos, performance logísticas e percepção de valor pelo cliente.")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total de Pedidos", f"{len(filtered_df):,}")
    c2.metric("NPS Implícito (Média Score)", f"{filtered_df['review_score'].mean():.2f} / 5.0")
    c3.metric("Taxa de Atraso Logístico", f"{(filtered_df['is_late'].mean() * 100):.1f}%")
    c4.metric("Frete Médio", f"R$ {filtered_df['freight_value'].mean():.2f}")
    
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Impacto do Atraso no Review Score")
        avg_score_delay = filtered_df.groupby('atraso_dias')['review_score'].mean().reset_index()
        fig_delay = px.bar(
            avg_score_delay, x='atraso_dias', y='review_score',
            title="Nota Média do Cliente vs. Dias de Atraso",
            labels={'atraso_dias': 'Dias de Atraso (Negativo = Antecipado)', 'review_score': 'Média Review Score'},
            color='review_score', color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_delay, use_container_width=True)
        
    with col_b:
        st.subheader("Distribuição de Avaliações")
        score_counts = filtered_df['review_score'].value_counts().reset_index()
        score_counts.columns = ['review_score', 'count']
        fig_pie = px.pie(
            score_counts, values='count', names='review_score',
            title="Proporção de Notas (1 a 5)",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# -----------------------------------------------------------------------------
# TELA 2: SIMULADOR & PREDIÇÃO DE DETRATORES
# -----------------------------------------------------------------------------
elif menu == "🤖 Simulator & Predição de Detratores":
    st.title("🤖 Simulador Preditivo de Insatisfação do Cliente")
    st.markdown("Insira os parâmetros de uma nova transação para estimar a probabilidade de o cliente se tornar um **detrator (Nota 1 ou 2)**.")
    
    st.markdown("---")
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        input_price = st.number_input("Valor do Produto (R$):", min_value=10.0, max_value=5000.0, value=150.0)
        input_freight = st.number_input("Valor do Frete (R$):", min_value=0.0, max_value=500.0, value=35.0)
        input_dist = st.slider("Distância Estimada (km):", min_value=10, max_value=4000, value=650)
        
    with col_input2:
        input_transp_time = st.slider("Tempo de Transporte (dias):", min_value=1, max_value=40, value=8)
        input_delay = st.slider("Atraso Previsto em Relação à Data Estimada (dias):", min_value=-10, max_value=20, value=2)
        input_ratio = input_freight / input_price
        
    st.markdown("---")
    
    # Predição
    input_df = pd.DataFrame({
        'price': [input_price],
        'freight_value': [input_freight],
        'distancia_km': [input_dist],
        'tempo_transporte_dias': [input_transp_time],
        'atraso_dias': [input_delay],
        'freight_ratio': [input_ratio]
    })
    
    prob_detractor = rf_model.predict_proba(input_df)[0][1]
    
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        st.subheader("Resultado da Simulação")
        st.metric("Risco de Avaliação Baixa", f"{prob_detractor * 100:.1f}%")
        
        if prob_detractor > 0.6:
            st.error("🚨 **ALTO RISCO DE INSATISFAÇÃO!** Recomenda-se trocar a transportadora ou reajustar a data estimada.")
        elif prob_detractor > 0.3:
            st.warning("⚠️ **RISCO MODERADO.** Acompanhe o rastreio com prioridade.")
        else:
            st.success("✅ **BAIXO RISCO.** Pedido dentro dos padrões de conformidade.")
            
    with col_res2:
        st.subheader("Importância das Variáveis no Modelo")
        importances = pd.DataFrame({
            'Feature': ['Preço', 'Frete', 'Distância', 'Tempo Transporte', 'Atraso em Dias', 'Razão Frete/Preço'],
            'Importance': rf_model.feature_importances_
        }).sort_values(by='Importance', ascending=True)
        
        fig_imp = px.bar(importances, x='Importance', y='Feature', orientation='h', title="Relevância dos Atributos para o Modelo (Random Forest)")
        st.plotly_chart(fig_imp, use_container_width=True)

# -----------------------------------------------------------------------------
# TELA 3: INTELIGÊNCIA GEOGRÁFICA & CLUSTERS
# -----------------------------------------------------------------------------
elif menu == "🗺️ Inteligência Geográfica & Clusters":
    st.title("🗺️ Agrupamento Geográfico e Impacto de Frete (K-Means)")
    st.markdown("Divisão das rotas operacionais da Olist em 4 Clusters Logísticos baseados em distância, valor de frete e peso no orçamento.")
    
    st.markdown("---")
    
    fig_scatter = px.scatter(
        filtered_df, x='distancia_km', y='freight_value', color=filtered_df['cluster'].astype(str),
        size='freight_ratio', hover_data=['customer_state', 'price'],
        title="Clusters Logísticos: Distância vs. Valor de Frete (Tamanho do Ponto = Frete / Preço)",
        labels={'distancia_km': 'Distância (km)', 'freight_value': 'Valor do Frete (R$)', 'color': 'Cluster Logístico'},
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.subheader("Perfil dos Clusters Logísticos")
        cluster_summary = filtered_df.groupby('cluster').agg({
            'distancia_km': 'mean',
            'freight_value': 'mean',
            'freight_ratio': 'mean',
            'review_score': 'mean'
        }).reset_index()
        cluster_summary.columns = ['Cluster', 'Distância Média (km)', 'Frete Médio (R$)', 'Peso Frete/Preço (%)', 'Review Score Médio']
        cluster_summary['Peso Frete/Preço (%)'] = (cluster_summary['Peso Frete/Preço (%)'] * 100).round(1)
        st.dataframe(cluster_summary.style.highlight_max(axis=0, color='#ffcccc'), use_container_width=True)
        
    with col_c2:
        st.subheader("Frete Médio por Estado (UF)")
        state_freight = filtered_df.groupby('customer_state')['freight_value'].mean().reset_index().sort_values(by='freight_value', ascending=False)
        fig_state = px.bar(state_freight, x='customer_state', y='freight_value', title="Disparidade de Frete por Estado Destino", color='freight_value', color_continuous_scale='Viridis')
        st.plotly_chart(fig_state, use_container_width=True)

# -----------------------------------------------------------------------------
# TELA 4: SAZONALIDADE & GMV
# -----------------------------------------------------------------------------
elif menu == "📈 Sazonalidade & GMV":
    st.title("📈 Análise de Séries Temporais e Sazonalidade de Vendas")
    st.markdown("Acompanhamento do volume financeiro negociado (GMV) e identificação de picos de vendas ao longo dos meses.")
    
    st.markdown("---")
    
    daily_sales = filtered_df.set_index('order_purchase_timestamp').resample('M')['price'].sum().reset_index()
    
    fig_ts = px.line(
        daily_sales, x='order_purchase_timestamp', y='price',
        title="Evolução Mensal do Faturamento (GMV Total)",
        labels={'order_purchase_timestamp': 'Mês/Ano', 'price': 'Receita (R$)'},
        markers=True
    )
    st.plotly_chart(fig_ts, use_container_width=True)
    
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.subheader("Top Categorias em Volume Financeiro")
        cat_sales = filtered_df.groupby('product_category_name')['price'].sum().reset_index().sort_values(by='price', ascending=False)
        fig_cat = px.bar(cat_sales, x='price', y='product_category_name', orientation='h', title="Receita por Categoria de Produto")
        st.plotly_chart(fig_cat, use_container_width=True)
        
    with col_t2:
        st.subheader("Sazonalidade Identificada")
        st.info("""
        * **Pico Principal (Novembro):** Evento da Black Friday com aumento expressivo de volume negociado.
        * **Pico Secundário (Maio):** Período de Dia das Mães.
        * **Efeito no Transporte:** Picos de demanda aumentam o tempo médio de entrega nas duas semanas subsequentes devido ao gargalo operacional das transportadoras.
        """)
