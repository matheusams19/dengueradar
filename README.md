# 🦟 DengueRadar | Plataforma de Inteligência Epidemiológica

> **CitizenTech | FIAP Global Solution 2026**

O **DengueRadar** é uma plataforma avançada de inteligência epidemiológica e monitoramento preditivo de surtos de dengue. Utilizando Machine Learning, Ciência de Dados e dados públicos oficiais (como SINAN/DATASUS e INMET), a solução traduz registros históricos em insights ativos para apoiar gestores de saúde pública na tomada de decisão preditiva.

---

## 👥 Autores
* **Matheus Alexandre**
* **Sabrina Fraguela**

---

## 🚀 Sobre o Projeto
O projeto foi desenvolvido no contexto da **FIAP Global Solution 2026**, integrando tecnologia, inovação social e rigor científico. Diferente de sistemas tradicionais baseados apenas em análise descritiva retrospectiva, o DengueRadar utiliza um modelo de **Random Forest** para antecipar o comportamento epidemiológico com base em variáveis climáticas e temporais.

### 🌟 Principais Funcionalidades
1. **Visão Geral Epidemiológica:** Painel executivo com métricas preditivas, variação percentual, incidência por 100 mil habitantes, nível de risco dinâmico e gráficos de tendência reais vs. previstos.
2. **Radar Geográfico (Inteligência Territorial):** Distribuição espacial interativa de focos e clusters de vulnerabilidade estimados com base em pesos populacionais e regionais, além de um ranking de intervenção prioritária exportável em CSV.
3. **Simulador Climático (Cenários A vs. B):** Ferramenta interativa de testes hipotéticos para avaliar o impacto de alterações na temperatura média e na precipitação acumulada nas projeções de surtos.
4. **Diagnóstico do Modelo (XAI):** Seção de transparência técnica com métricas reais de desempenho (MAE, RMSE), gráfico de importância das variáveis (*Feature Importance*) e análise de resíduos.
5. **Metodologia e Transparência:** Documentação científica detalhada, explicitando as fontes de dados, o pipeline preditivo e as limitações operacionais da solução.

---

## 🛠️ Stack Tecnológica
* **Linguagem:** Python 3.10+
* **Interface Web:** Streamlit (Layout Customizado em Dark Mode - *Deep Navy*)
* **Modelagem Preditiva:** Scikit-Learn (`RandomForestRegressor`)
* **Visualização de Dados:** Plotly & Streamlit Native Maps
* **Manipulação de Dados:** Pandas & NumPy

---

## 📦 Estrutura do Repositório
```text
dengueradar-fiap/
│
├── app.py                     # Aplicação principal em Streamlit (Multi-page)
├── requirements.txt           # Dependências do projeto
├── LogoDengueRadar.png        # Logótipo oficial da plataforma
├── models/
│   ├── modelo_rf_dengue.pkl   # Modelo de Machine Learning treinado
│   └── features_dengue.pkl    # Lista de features do modelo
├── data/
│   └── processed/             # Datasets processados
└── notebooks/
    ├── 01_limpeza_de_dados.ipynb # Pipeline de ETL e tratamento
    └── 02_modelo_ia.ipynb        # Treinamento, backtesting e salvamento da IA
