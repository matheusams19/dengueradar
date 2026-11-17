import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import os

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA E CSS
# ==========================================
st.set_page_config(
    page_title="DengueRadar | CitizenTech",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    
    <style>
    /* Tema Escuro Profissional (Deep Navy) */
    .stApp { background-color: #0B1120; color: #F8FAFC; font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4 { font-weight: 600; color: #FFFFFF; }
    .subtitle { font-size: 1.1rem; color: #94A3B8; margin-top: -15px; margin-bottom: 25px; }
    
    /* Cards de Métricas Estilo Dashboard Executivo */
    div[data-testid="metric-container"] {
        background-color: #1E293B;
        padding: 20px 20px;
        border-radius: 12px;
        border-top: 4px solid #38BDF8;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.15);
    }
    div[data-testid="metric-container"] > label { color: #94A3B8 !important; font-size: 1rem; font-weight: 500;}
    div[data-testid="metric-container"] > div { color: #F8FAFC !important; font-size: 2.2rem !important; font-weight: 700;}
    
    /* Alertas customizados */
    .alert-card { padding: 15px; border-radius: 8px; margin-bottom: 20px; font-weight: 500; border-left: 5px solid; }
    .alert-baixo { background-color: rgba(16, 185, 129, 0.1); border-color: #10B981; color: #10B981; }
    .alert-medio { background-color: rgba(245, 158, 11, 0.1); border-color: #F59E0B; color: #F59E0B; }
    .alert-alto { background-color: rgba(239, 68, 68, 0.1); border-color: #EF4444; color: #EF4444; }
    .alert-critico { background-color: rgba(153, 27, 27, 0.2); border-color: #991B1B; color: #FCA5A5; font-weight: 700;}
    
    /* Ajuste de separadores */
    hr { border-color: #1E293B; margin-top: 2rem; margin-bottom: 2rem; }
    </style>

""", unsafe_allow_html=True)


# ==========================================
# 2. CARREGAMENTO DE DADOS E MODELO (CACHE)
# ==========================================
@st.cache_resource
def load_model():
    try:
        modelo = joblib.load('models/modelo_rf_dengue.pkl')
        features = joblib.load('models/features_dengue.pkl')
        return modelo, features
    except Exception as e:
        return None, None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/processed/dataset_dengueradar_limpo.csv')
        return df, True
    except:
        anos = [2023, 2024]
        semanas = list(range(1, 53))
        dados = []
        for ano in anos:
            for sem in semanas:
                temp = 20 + 10 * np.sin(sem * np.pi / 26) + np.random.normal(0, 2)
                chuva = max(0, 100 + 80 * np.sin(sem * np.pi / 26) + np.random.normal(0, 30))
                casos = max(0, int((temp * 2.5) + (chuva * 0.8) + np.random.normal(0, 50)))
                dados.append({'ano': ano, 'semana_epidem': sem, 'temp_media': temp, 'precipitacao_total': chuva, 'casos': casos})
        return pd.DataFrame(dados), False

modelo_rf, features_cols = load_model()
df_historico, dados_reais = load_data()


# ==========================================
# 3. FUNÇÕES AUXILIARES DE NEGÓCIO
# ==========================================
def calcular_risco(casos_previstos, incidencia):
    if casos_previstos < 200 and incidencia < 100:
        return "BAIXO", "alert-baixo", "#10B981"
    elif casos_previstos < 600 and incidencia < 300:
        return "MODERADO", "alert-medio", "#F59E0B"
    elif casos_previstos < 1200:
        return "ALTO", "alert-alto", "#EF4444"
    else:
        return "CRÍTICO", "alert-critico", "#991B1B"

def gerar_recomendacoes(risco):
    recom = {
        "BAIXO": ["• Manutenção da vigilância e monitoramento de indicadores.", "• Campanhas educativas e preventivas contínuas."],
        "MODERADO": ["• Intensificação de ações preventivas (ex: vistoria de focos).", "• Revisão da disponibilidade de recursos nas UBS."],
        "ALTO": ["• Priorização da vigilância em áreas críticas com fumacê espacial.", "• Avaliação imediata da capacidade de leitos."],
        "CRÍTICO": ["• Comunicação imediata do risco aos gestores e Defesa Civil.", "• Ativação de plano de contingência e abertura de leitos de hidratação."]
    }
    return recom.get(risco, [])

def simular_distribuicao_espacial(casos_totais):
    # Populações municipais reais (IBGE/SEADE) para cálculo correto e proporcional da incidência
    municipios = [
        {"mun": "São Paulo", "lat": -23.5505, "lon": -46.6333, "peso": 0.45, "pop": 11451999},
        {"mun": "Guarulhos", "lat": -23.4628, "lon": -46.5333, "peso": 0.15, "pop": 1291771},
        {"mun": "Campinas", "lat": -22.9099, "lon": -47.0626, "peso": 0.12, "pop": 1138309},
        {"mun": "Osasco", "lat": -23.5329, "lon": -46.7917, "peso": 0.10, "pop": 699944},
        {"mun": "Santo André", "lat": -23.6638, "lon": -46.5383, "peso": 0.10, "pop": 748919},
        {"mun": "São Bernardo do Campo", "lat": -23.6939, "lon": -46.5644, "peso": 0.08, "pop": 810729}
    ]
    dados = []
    for m in municipios:
        casos_estimados = int(casos_totais * m["peso"])
        incidencia_est = (casos_estimados / m["pop"]) * 100000
        dados.append({
            "Município/Região": m["mun"],
            "Casos Previstos": casos_estimados,
            "Incidência (100k)": round(incidencia_est, 1),
            "Lat": m["lat"],
            "Lon": m["lon"]
        })
    return pd.DataFrame(dados).sort_values("Casos Previstos", ascending=False)

def gerar_pontos_individuais(df_resumo):
    pontos = []
    for _, row in df_resumo.iterrows():
        n_casos = int(row['Casos Previstos'])
        if n_casos > 0:
            lats = np.random.normal(loc=row['Lat'], scale=0.035, size=n_casos)
            lons = np.random.normal(loc=row['Lon'], scale=0.035, size=n_casos)
            for lat, lon in zip(lats, lons):
                pontos.append({
                    'Município': row['Município/Região'],
                    'lat': lat,
                    'lon': lon
                })
    return pd.DataFrame(pontos)


# ==========================================
# 4. BARRA LATERAL (MENU E CONTROLES)
# ==========================================
with st.sidebar:
    st.image("LogoDengueRadar.png", use_container_width=True)
    st.markdown("---")
    
    menu = st.radio("Navegação Principal", [
        "1. Visão Geral Epidemiológica",
        "2. Inteligência Territorial (Radar)",
        "3. Simulador e Cenários",
        "4. Diagnóstico do Modelo",
        "5. Dados e Metodologia"
    ])
    
    st.markdown("---")
    if not dados_reais:
        st.warning("⚠️ **Aviso:** Utilizando dados de fallback (simulados) pois o CSV original não foi encontrado.")
        
    st.caption("Desenvolvido para FIAP | CitizenTech")
    st.caption("✨ Versão 2.0 (Deep Navy) - Atualizada")


# ==========================================
# 5. PÁGINA 1: VISÃO GERAL EPIDEMIOLÓGICA
# ==========================================
if menu == "1. Visão Geral Epidemiológica":
    st.title("Visão Geral Epidemiológica")
    st.markdown('<p class="subtitle">Monitoramento preditivo e situação atual de surtos endêmicos.</p>', unsafe_allow_html=True)
    
    if modelo_rf is None:
        st.error("Modelo de IA não treinado. Acesse o Notebook 02 e gere os arquivos `.pkl` na pasta `models/`.")
        st.stop()
        
    col_a, col_b, col_c = st.columns(3)
    semana_atual = df_historico['semana_epidem'].max() if not df_historico.empty else 15
    with col_a: semana_sel = st.slider("Selecione a Semana Epidemiológica", 1, 52, int(semana_atual))
    
    hist_semana = df_historico[df_historico['semana_epidem'] == semana_sel]
    temp_ref = hist_semana['temp_media'].mean() if not hist_semana.empty else 25.0
    chuva_ref = hist_semana['precipitacao_total'].mean() if not hist_semana.empty else 50.0
    
    input_data = {'semana_epidem': semana_sel, 'temp_media': temp_ref, 'precipitacao_total': chuva_ref}
    input_df = pd.DataFrame([input_data])[features_cols]
    
    previsao = modelo_rf.predict(input_df)[0]
    casos_ultimo_periodo = hist_semana['casos'].mean() if not hist_semana.empty else 0
    variacao = ((previsao - casos_ultimo_periodo) / casos_ultimo_periodo) * 100 if casos_ultimo_periodo > 0 else 0
    
    incidencia = (previsao / 12000000) * 100000
    risco_label, risco_class, risco_cor = calcular_risco(previsao, incidencia)
    
    st.markdown("### Indicadores de Monitoramento")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Casos Previstos", f"{int(previsao)}", f"{variacao:+.1f}% vs obs.", delta_color="inverse")
    m2.metric("Casos Observados (Histórico)", f"{int(casos_ultimo_periodo)}")
    m3.metric("Incidência (por 100k hab)", f"{incidencia:.1f}")
    m4.metric("Nível de Risco", risco_label)
    m5.metric("Temperatura Referência", f"{temp_ref:.1f}°C")
    
    st.markdown(f'<div class="alert-card {risco_class}"><strong>STATUS DO ALERTA:</strong> Risco {risco_label}. Ação sugerida: {gerar_recomendacoes(risco_label)[0].replace("•", "").strip()}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    c_graf1, c_graf2 = st.columns([2, 1])
    
    with c_graf1:
        st.markdown("#### Casos Reais vs. Previstos (Tendência)")
        df_plot = df_historico.groupby('semana_epidem')['casos'].mean().reset_index()
        df_plot['casos_previstos'] = modelo_rf.predict(df_historico.groupby('semana_epidem')[features_cols].mean())
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot['semana_epidem'], y=df_plot['casos'], mode='lines', name='Casos Reais (Observados)', line=dict(color='#38BDF8', width=3)))
        fig.add_trace(go.Scatter(x=df_plot['semana_epidem'], y=df_plot['casos_previstos'], mode='lines', name='Projeção do Modelo', line=dict(color='#F43F5E', width=3, dash='dash')))
        
        std_dev = df_plot['casos_previstos'] * 0.15
        fig.add_trace(go.Scatter(x=pd.concat([df_plot['semana_epidem'], df_plot['semana_epidem'][::-1]]),
                                 y=pd.concat([df_plot['casos_previstos'] + std_dev, (df_plot['casos_previstos'] - std_dev)[::-1]]),
                                 fill='toself', fillcolor='rgba(244, 63, 94, 0.08)', line=dict(color='rgba(255,255,255,0)'),
                                 hoverinfo="skip", showlegend=True, name='Faixa de Incerteza (Ilustrativa)'))
        
        fig.add_vline(x=semana_sel, line_width=2, line_dash="dot", line_color="white", annotation_text="Semana Selecionada")
        fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                          font=dict(color='#F8FAFC'), hovermode="x unified",
                          xaxis=dict(showgrid=False), yaxis=dict(gridcolor='#334155'))
        st.plotly_chart(fig, use_container_width=True)
        
    with c_graf2:
        st.markdown("#### Escala de Risco Epidemiológico")
        limite_max = max(1500, previsao * 1.25)
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = previsao,
            gauge = {
                'axis': {'range': [0, limite_max], 'tickwidth': 1, 'tickcolor': "#CBD5E1"},
                'bar': {'color': risco_cor}, 'bgcolor': "#1E293B", 'borderwidth': 0,
                'steps': [
                    {'range': [0, 200], 'color': 'rgba(16, 185, 129, 0.15)'},
                    {'range': [200, 600], 'color': 'rgba(245, 158, 11, 0.15)'},
                    {'range': [600, limite_max], 'color': 'rgba(239, 68, 68, 0.15)'}],
            }
        ))
        fig_gauge.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#F8FAFC'), margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    st.markdown("### 📋 Plano de Ação Recomendado")
    rec_list = gerar_recomendacoes(risco_label)
    for r in rec_list:
        st.write(r)
    st.caption("*Nota: O DengueRadar é um sistema de apoio à decisão. Medidas operacionais devem ser validadas com a Vigilância Sanitária local.*")

    st.markdown("---")
    st.subheader("📥 Exportação de Dados")
    
    df_export = pd.DataFrame({
        "Semana Epidemiológica": [semana_sel],
        "Casos Previstos": [int(previsao)],
        "Casos Observados (Histórico)": [int(casos_ultimo_periodo)],
        "Incidência (por 100k hab)": [round(incidencia, 1)],
        "Nível de Risco Estimado": [risco_label],
        "Temperatura Média (°C)": [round(temp_ref, 1)],
        "Precipitação (mm)": [round(chuva_ref, 1)]
    })
    
    csv_overview = df_export.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Resumo Epidemiológico (CSV)",
        data=csv_overview,
        file_name=f'resumo_epidemiologico_semana_{semana_sel}.csv',
        mime='text/csv',
    )


# ==========================================
# 6. PÁGINA 2: INTELIGÊNCIA TERRITORIAL
# ==========================================
elif menu == "2. Inteligência Territorial (Radar)":
    st.title("Radar Geográfico Epidemiológico")
    st.markdown('<p class="subtitle">Análise espacial de risco e priorização territorial com distribuição heurística por casos.</p>', unsafe_allow_html=True)
    
    casos_base = st.sidebar.number_input("Casos Globais (Simulador Mapa)", value=1500, step=100)
    df_espacial = simular_distribuicao_espacial(casos_base)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.write("Filtros de Região")
        opcoes_cidades = ["Todos os Municípios"] + list(df_espacial['Município/Região'].unique())
        cidade_sel = st.selectbox("Filtrar Município Específico", opcoes_cidades)
        
    with col2:
        st.info(f"💡 **Inteligência Territorial:** Cada pontinho no mapa representa visualmente **1 caso estimado** distribuído heuristicamente em torno do centroide populacional do município.")
    
    st.markdown("---")
    c_mapa, c_rank = st.columns([2, 1.2])
    
    if cidade_sel != "Todos os Municípios":
        df_espacial_filtrado = df_espacial[df_espacial['Município/Região'] == cidade_sel]
    else:
        df_espacial_filtrado = df_espacial
        
    df_pontinhos = gerar_pontos_individuais(df_espacial_filtrado)
    
    with c_mapa:
        st.markdown(f"#### Distribuição Espacial Simulada: {cidade_sel}")
        if not df_pontinhos.empty:
            st.map(df_pontinhos, latitude="lat", longitude="lon", color="#EF4444")
        else:
            st.warning("Nenhum caso projetado para esta seleção.")
            
        st.caption(f"Visualizando {len(df_pontinhos)} pontos de distribuição estimada. Aviso: Coordenadas dispersadas estatisticamente num raio aproximado de 4km do centro (heurística de visualização).")
        
    with c_rank:
        st.markdown("#### 🚨 Ranking de Intervenção")
        df_rank = df_espacial.copy()
        df_rank['Nível de Risco'] = df_rank['Casos Previstos'].apply(lambda x: "CRÍTICO" if x > 400 else "ALTO" if x > 150 else "MODERADO")
        
        if cidade_sel != "Todos os Municípios":
            st.dataframe(df_rank[df_rank['Município/Região'] == cidade_sel][['Município/Região', 'Casos Previstos', 'Incidência (100k)', 'Nível de Risco']], 
                         use_container_width=True, hide_index=True)
            st.markdown("Outras regiões:")
            st.dataframe(df_rank[df_rank['Município/Região'] != cidade_sel][['Município/Região', 'Casos Previstos', 'Incidência (100k)', 'Nível de Risco']], 
                         use_container_width=True, hide_index=True)
        else:
            st.dataframe(df_rank[['Município/Região', 'Casos Previstos', 'Incidência (100k)', 'Nível de Risco']], 
                         use_container_width=True, hide_index=True)
        
        csv = df_rank.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Exportar Ranking (CSV)", data=csv, file_name='ranking_regioes.csv', mime='text/csv')


# ==========================================
# 7. PÁGINA 3: SIMULADOR E CENÁRIOS
# ==========================================
elif menu == "3. Simulador e Cenários":
    st.title("Simulador Climático e Comparação de Cenários")
    st.markdown('<p class="subtitle">Avalie o impacto hipotético de mudanças na temperatura e precipitação nas projeções.</p>', unsafe_allow_html=True)
    
    st.write("Defina um cenário de referência e ajuste os parâmetros no cenário simulado para verificar o delta projetado.")
    
    col_ref, col_sim = st.columns(2)
    
    with col_ref:
        st.markdown("<div class='alert-card' style='background-color:#1E293B;'>", unsafe_allow_html=True)
        st.subheader("Cenário A (Referência)")
        sem_a = st.number_input("Semana", value=15, key='sem_a')
        temp_a = st.number_input("Temperatura (°C)", value=25.0, key='temp_a')
        chuva_a = st.number_input("Precipitação (mm)", value=50.0, key='chuva_a')
        
        in_a = pd.DataFrame([{'semana_epidem': sem_a, 'temp_media': temp_a, 'precipitacao_total': chuva_a}])[features_cols]
        prev_a = modelo_rf.predict(in_a)[0]
        st.metric("Casos Previstos (A)", int(prev_a))
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_sim:
        st.markdown("<div class='alert-card' style='background-color:#1E293B;'>", unsafe_allow_html=True)
        st.subheader("Cenário B (Simulado)")
        sem_b = st.number_input("Semana", value=15, key='sem_b')
        temp_b = st.slider("Temperatura Ajustada (°C)", min_value=15.0, max_value=40.0, value=32.0, key='temp_b')
        chuva_b = st.slider("Precipitação Ajustada (mm)", min_value=0.0, max_value=300.0, value=150.0, key='chuva_b')
        
        in_b = pd.DataFrame([{'semana_epidem': sem_b, 'temp_media': temp_b, 'precipitacao_total': chuva_b}])[features_cols]
        prev_b = modelo_rf.predict(in_b)[0]
        
        delta_val = prev_b - prev_a
        delta_pct = (delta_val / prev_a * 100) if prev_a > 0 else 0
        st.metric("Casos Previstos (B)", int(prev_b), f"{delta_pct:+.1f}% de diferença", delta_color="inverse")
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("#### Impacto da Simulação")
    
    fig_comp = go.Figure(data=[
        go.Bar(name='Cenário A', x=['Casos Estimados'], y=[prev_a], marker_color='#38BDF8'),
        go.Bar(name='Cenário B', x=['Casos Estimados'], y=[prev_b], marker_color='#F43F5E')
    ])
    fig_comp.update_layout(barmode='group', height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#F8FAFC'))
    st.plotly_chart(fig_comp, use_container_width=True)
    
    st.info("⚠️ Mudanças climáticas extremas fora da distribuição dos dados de treinamento podem gerar previsões menos confiáveis (extrapolação matemática).")


# ==========================================
# 8. PÁGINA 4: DIAGNÓSTICO DO MODELO
# ==========================================
elif menu == "4. Diagnóstico do Modelo":
    st.title("Diagnóstico e Transparência do Modelo (XAI)")
    st.markdown('<p class="subtitle">Auditoria técnica do modelo de Machine Learning em produção.</p>', unsafe_allow_html=True)
    
    if modelo_rf is None:
        st.warning("Modelo não carregado.")
        st.stop()
        
    X = df_historico[features_cols]
    y_true = df_historico['casos']
    y_pred = modelo_rf.predict(X)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Modelo Base", "RandomForest", "Scikit-Learn")
    col2.metric("Registros Avaliados", len(df_historico))
    col3.metric("MAE (In-Sample)", f"{mae:.1f} casos")
    col4.metric("RMSE (In-Sample)", f"{rmse:.1f}")
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### Explicabilidade: Importância das Variáveis")
        importances = modelo_rf.feature_importances_
        df_imp = pd.DataFrame({'Variável': features_cols, 'Importância': importances}).sort_values('Importância', ascending=True)
        
        fig_imp = px.bar(df_imp, x='Importância', y='Variável', orientation='h', color_discrete_sequence=['#38BDF8'])
        fig_imp.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#F8FAFC'))
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption("O gráfico indica quais fatores climáticos e temporais exercem maior peso matemático na decisão de projeção de casos da IA.")
        
    with c2:
        st.markdown("#### Análise de Resíduos (Erro de Predição)")
        residuos = y_true - y_pred
        fig_res = px.scatter(x=y_pred, y=residuos, labels={'x':'Casos Previstos', 'y':'Resíduo (Real - Previsto)'}, color_discrete_sequence=['#F43F5E'])
        fig_res.add_hline(y=0, line_dash="dash", line_color="white")
        fig_res.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#F8FAFC'))
        fig_res.update_xaxes(gridcolor='#334155')
        fig_res.update_yaxes(gridcolor='#334155')
        st.plotly_chart(fig_res, use_container_width=True)
        st.caption("Verificação de viés: Idealmente, os pontos devem estar distribuídos aleatoriamente em torno da linha zero.")


# ==========================================
# 9. PÁGINA 5: DADOS E METODOLOGIA
# ==========================================
elif menu == "5. Dados e Metodologia":
    st.title("Metodologia Científica e Documentação")
    st.markdown('<p class="subtitle">Como o DengueRadar processa dados públicos para apoiar a saúde.</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Objetivo da Plataforma
    O **DengueRadar** é uma plataforma de inteligência epidemiológica desenvolvida no contexto da FIAP (CitizenTech). O sistema integra Ciência de Dados e dados públicos para auxiliar gestores na antecipação de surtos e planejamento de ações, traduzindo registros históricos em insights preditivos.

    ### 📊 Fontes de Dados
    * **SINAN (Sistema de Informação de Agravos de Notificação):** Dados oficiais de casos notificados, acessados via plataforma DATASUS/TABNET.
    * **INMET (Instituto Nacional de Meteorologia):** Séries históricas de variáveis climáticas (temperatura e precipitação), agregadas na mesma janela temporal.

    ### 🧠 Modelagem e Inteligência Artificial
    O motor preditivo atual baseia-se num algoritmo **Random Forest Regressor** (Scikit-Learn).
    * **Justificativa:** O modelo consegue mapear interações não-lineares, essenciais para compreender como limiares combinados de chuva e calor desencadeiam ciclos de proliferação do mosquito.
    * **Validação:** As métricas de avaliação avaliam o ajuste atual, servindo como base para futuras implementações de validação temporal estrita (*walk-forward*).

    ### ⚠️ Confiabilidade e Limitações
    A transparência é um pilar do DengueRadar. É fundamental compreender as ressalvas da plataforma:
    1. **Atraso de Notificação:** Os dados observados sofrem atrasos institucionais. O modelo trata dados históricos como consolidados, o que pode não refletir a realidade em tempo real.
    2. **Distribuição Espacial Heurística:** A aba "Inteligência Territorial" distribui as previsões globais utilizando ponderação populacional com base em dados reais do IBGE/SEADE. Os pontos representam áreas de **vulnerabilidade teórica estimada**, e não localizações exatas de pacientes.
    3. **Extrapolação Climática:** Simulações com parâmetros climáticos muito além do histórico de treinamento podem gerar previsões estatisticamente não confiáveis.
    4. **Sistema de Apoio:** A IA não toma decisões clínicas. As projeções apoiam o planejamento, mas as ações devem seguir os protocolos oficiais da Vigilância em Saúde.
""")
    
    st.markdown("---")
    st.info("Desenvolvido por alunos da FIAP - Matheus Alexandre e Sabrina Fraguela para integrar inovação social, tecnologia preditiva e inteligência cidadã.")