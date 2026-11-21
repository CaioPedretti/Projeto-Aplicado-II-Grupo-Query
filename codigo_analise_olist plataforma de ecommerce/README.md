# Scripts de Análise e Protótipo do Dashboard

Os códigos desta pasta estão divididos de acordo com a sua função no projeto:

## 1. Scripts de Modelagem nos Dados Reais (CSVs Olist - Kaggle)
Todas as métricas apresentadas no relatório técnico (Acurácia, F1-Score, ROC-AUC, Baseline Ingênuo, Odds Ratio, Silhueta, MAPE e WAPE) foram geradas a partir dos arquivos CSV reais do conjunto público da Olist utilizando os três scripts abaixo:

* `avaliacao_a3_olist.py`: Treinamento e validação dos modelos de classificação (Random Forest, Regressão Logística e Baseline Ingênuo) com divisão 70/15/15 sem vazamento de dados, além da modelagem de séries temporais (STL e SARIMAX) com holdout temporal de 6 meses.
* `avaliacao_kmeans_olist.py`: Cálculo das distâncias entre vendedores e clientes pela fórmula de Haversine e agrupamento K-Means sobre os itens reais entregues.
* `analises_complementares_olist.py`: Testes estatísticos de impacto do atraso nas avaliações (Odds Ratio na base completa, recorte de atrasos severos até 30 dias, regressão linear e correlação de Spearman).

## 2. Protótipo do Produto Interativo (Dashboard)
* `app.py`: Painel interativo construído em Streamlit e Plotly (`Olist Delivery & Satisfaction Intelligence`). Para permitir a abertura e a demonstração rápida da interface sem exigir o download local dos CSVs brutos do Kaggle, o aplicativo gera uma amostra em memória no carregamento inicial.
