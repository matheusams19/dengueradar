import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import os
import json
from pathlib import Path

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
# Os artefatos são carregados a partir da pasta do projeto, independentemente do diretório de execução.
BASE = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    try:
        modelo = joblib.load(BASE / 'models/modelo_rf_dengue.pkl')
        features = joblib.load(BASE / 'models/features_dengue.pkl')
        return modelo, features, None
    except (FileNotFoundError, OSError, ValueError, ImportError) as e:
        return None, None, str(e)

@st.cache_data
def load_data():
    caminho = BASE / 'data/processed/dataset_dengueradar_limpo.csv'
    if not caminho.exists():
        raise FileNotFoundError(f'Dados reais indisponíveis: {caminho}. Execute o notebook 01 antes de publicar.')
    df = pd.read_csv(caminho)
    necessarias = {'ano', 'semana_epidem', 'casos', 'temp_media', 'precipitacao_total'}
    if not necessarias.issubset(df.columns):
        raise ValueError(f'Colunas ausentes no CSV: {sorted(necessarias - set(df.columns))}')
    if df[list(necessarias)].isna().any().any() or df.duplicated(['ano','semana_epidem']).any():
        raise ValueError('CSV contém ausências ou semanas duplicadas. Reexecute a limpeza e validação.')
    return df.sort_values(['ano','semana_epidem']).reset_index(drop=True)

@st.cache_data
def load_evaluation():
    pasta = BASE / 'models'
    meta_path, pred_path = pasta / 'avaliacao_modelo.json', pasta / 'avaliacao_temporal.csv'
    meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else None
    pred = pd.read_csv(pred_path) if pred_path.exists() else None
    return meta, pred

modelo_rf, features_cols, erro_modelo = load_model()
try:
    df_historico = load_data()
except (FileNotFoundError, ValueError, pd.errors.ParserError) as erro:
    st.error(f'Não foi possível abrir o histórico real: {erro}')
    st.info('Por segurança, o DengueRadar não substitui a base epidemiológica por registros simulados.')
    st.stop()

if modelo_rf is None:
    st.error(f'Modelo indisponível: {erro_modelo}. Execute o notebook 02 e envie os artefatos à pasta models/.')
    st.stop()
if not features_cols or any(col not in df_historico.columns for col in features_cols):
    st.error('Features do modelo incompatíveis com a base de dados.')
    st.stop()
meta_avaliacao, df_avaliacao = load_evaluation()

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
        "ALTO": ["• Priorização de inspeções e ações de controle conforme protocolos locais.", "• Avaliação imediata da capacidade de leitos."],
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
    # Amostra determinística, limitada para evitar sobrecarregar o navegador. NÃO representa casos geolocalizados.
    rng = np.random.default_rng(42)
    pontos = []
    for _, row in df_resumo.iterrows():
        n_casos = min(1500, int(row['Casos Previstos']))
        if n_casos > 0:
            lats = rng.normal(loc=row['Lat'], scale=0.035, size=n_casos)
            lons = rng.normal(loc=row['Lon'], scale=0.035, size=n_casos)
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
    if (BASE / 'LogoDengueRadar.png').exists():
        st.image(str(BASE / 'LogoDengueRadar.png'), use_container_width=True)
    else:
        st.markdown('### 🦟 DengueRadar')
    st.markdown("---")
    
    menu = st.radio("Navegação Principal", [
        "1. Visão Geral Epidemiológica",
        "2. Inteligência Territorial (Radar)",
        "3. Simulador e Cenários",
        "4. Diagnóstico do Modelo",
        "5. Dados e Metodologia"
    ])
    
    st.markdown("---")
    st.caption("Base real carregada. Consulte a cobertura temporal e as limitações na página de metodologia.")
        
    st.caption("Desenvolvido para FIAP | CitizenTech")
    st.caption("Versão revisada — indicadores demonstrativos devidamente identificados")


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
    semana_atual = int(df_historico.iloc[-1]['semana_epidem'])
    with col_a: semana_sel = st.slider("Semana epidemiológica de referência (histórica)", 1, 53, int(semana_atual))
    
    hist_semana = df_historico[df_historico['semana_epidem'] == semana_sel]
    st.caption('Comparação retrospectiva por semana do ano, agregando anos distintos; não é previsão operacional da próxima semana.')
    temp_ref = hist_semana['temp_media'].mean() if not hist_semana.empty else 25.0
    chuva_ref = hist_semana['precipitacao_total'].mean() if not hist_semana.empty else 50.0
    
    input_data = {'semana_epidem': semana_sel, 'temp_media': temp_ref, 'precipitacao_total': chuva_ref}
    input_df = pd.DataFrame([input_data])[features_cols]
    
    previsao = modelo_rf.predict(input_df)[0]
    casos_ultimo_periodo = hist_semana['casos'].mean() if not hist_semana.empty else 0
    variacao = ((previsao - casos_ultimo_periodo) / casos_ultimo_periodo) * 100 if casos_ultimo_periodo > 0 else 0
    
    # O histórico não identifica um denominador populacional nem território de referência.
    # Não calcular incidência agregada com população arbitrária.
    incidencia = float('nan')
    # Classificação demonstrativa por volume absoluto, sem validade epidemiológica municipal.
    risco_label, risco_class, risco_cor = calcular_risco(previsao, 0)
    
    st.markdown("### Indicadores de Monitoramento")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Estimativa do modelo", f"{int(round(previsao))}", f"{variacao:+.1f}% vs média histórica", delta_color="inverse")
    m2.metric("Média histórica da semana", f"{casos_ultimo_periodo:.0f}")
    m3.metric("Incidência real", "N/D", help="Requer território e população compatíveis com os casos do DATASUS.")
    m4.metric("Faixa demonstrativa", risco_label)
    m5.metric("Temperatura Referência", f"{temp_ref:.1f}°C")
    
    st.markdown(f'<div class="alert-card {risco_class}"><strong>CLASSIFICAÇÃO EXPLORATÓRIA:</strong> Risco {risco_label}. Ação sugerida: {gerar_recomendacoes(risco_label)[0].replace("•", "").strip()}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    c_graf1, c_graf2 = st.columns([2, 1])
    
    with c_graf1:
        st.markdown("#### Médias históricas por semana do ano e ajuste do modelo")
        df_plot = df_historico.groupby('semana_epidem')['casos'].mean().reset_index()
        df_plot['casos_previstos'] = np.maximum(0, modelo_rf.predict(df_historico.groupby('semana_epidem')[features_cols].mean()[features_cols]))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot['semana_epidem'], y=df_plot['casos'], mode='lines', name='Casos Reais (Observados)', line=dict(color='#38BDF8', width=3)))
        fig.add_trace(go.Scatter(x=df_plot['semana_epidem'], y=df_plot['casos_previstos'], mode='lines', name='Ajuste do modelo (não é teste futuro)', line=dict(color='#F43F5E', width=3, dash='dash')))
        
        st.caption('Este gráfico compara médias de diferentes anos. Para avaliar generalização temporal, consulte o diagnóstico do modelo.')

        fig.add_vline(x=semana_sel, line_width=2, line_dash="dot", line_color="white", annotation_text="Semana Selecionada")
        fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                          font=dict(color='#F8FAFC'), hovermode="x unified",
                          xaxis=dict(showgrid=False), yaxis=dict(gridcolor='#334155'))
        st.plotly_chart(fig, use_container_width=True)
        
    with c_graf2:
        st.markdown("#### Volume estimado — escala exploratória")
        st.caption("Faixas em números absolutos apenas demonstrativas; não são limiares oficiais de alerta.")
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
        
    st.markdown("### 📋 Orientações demonstrativas (não prescritivas)")
    rec_list = gerar_recomendacoes(risco_label)
    for r in rec_list:
        st.write(r)
    st.caption("*Recomendações demonstrativas, não alertas oficiais. Ações de vigilância devem ser validadas pelas autoridades de Vigilância Epidemiológica.*")

    st.markdown("---")
    st.subheader("📥 Exportação de Dados")
    
    df_export = pd.DataFrame({
        "Semana Epidemiológica": [semana_sel],
        "Estimativa retrospectiva do modelo": [int(round(previsao))],
        "Média observada da semana ao longo dos anos": [round(casos_ultimo_periodo, 1)],
        "Incidência (por 100k hab)": [None],
        "Classificação Exploratória": [risco_label],
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
        st.info(f"💡 **Inteligência Territorial:** Cada ponto é um **marcador fictício de distribuição ilustrativa** e NÃO corresponde a um caso ou foco verdadeiro. Exibimos no máximo 1.500 pontos por município.")
    
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
            
        st.caption(f"Visualizando {len(df_pontinhos)} marcadores sintéticos, limitados por município. Não são registros georreferenciados nem locais de transmissão.")
        
    with c_rank:
        st.markdown("#### Distribuição heurística de casos (não é ranking validado)")
        df_rank = df_espacial.copy()
        df_rank['Nível de Risco'] = 'NÃO AVALIADO'
        st.caption('Casos são distribuídos por pesos fixos escolhidos para demonstração. A incidência ilustrativa usa populações municipais aproximadas e NÃO permite priorização epidemiológica.')
        
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
    st.markdown("#### Diferença estimada entre cenários hipotéticos")
    
    fig_comp = go.Figure(data=[
        go.Bar(name='Cenário A', x=['Casos Estimados'], y=[prev_a], marker_color='#38BDF8'),
        go.Bar(name='Cenário B', x=['Casos Estimados'], y=[prev_b], marker_color='#F43F5E')
    ])
    fig_comp.update_layout(barmode='group', height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#F8FAFC'))
    st.plotly_chart(fig_comp, use_container_width=True)
    
    st.info("⚠️ Este simulador compara respostas hipotéticas do modelo. Não demonstra causalidade nem estima surtos futuros com clima previsto. Mudanças fora da distribuição de treino podem ser pouco confiáveis.")


# ==========================================
# 8. PÁGINA 4: DIAGNÓSTICO DO MODELO
# ==========================================
elif menu == "4. Diagnóstico do Modelo":
    st.title("Diagnóstico e Transparência do Modelo (XAI)")
    st.markdown('<p class="subtitle">Auditoria técnica do modelo de Machine Learning em produção.</p>', unsafe_allow_html=True)
    
    if modelo_rf is None:
        st.warning("Modelo não carregado.")
        st.stop()
        
    if meta_avaliacao is None or df_avaliacao is None:
        st.warning('Avaliação temporal não encontrada. Execute o notebook 02 atualizado para gerar models/avaliacao_modelo.json e models/avaliacao_temporal.csv. Métricas de treinamento não substituem teste futuro.')
    else:
        mrf = meta_avaliacao.get('metricas_rf_teste', {})
        mbase = meta_avaliacao.get('metricas_baseline_teste', {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric('Modelo', 'RandomForest')
        col2.metric('Semanas de teste futuro', meta_avaliacao.get('registros_teste', 'N/D'))
        col3.metric('MAE fora da amostra', f"{mrf.get('MAE', 0):.1f} casos")
        col4.metric('RMSE fora da amostra', f"{mrf.get('RMSE', 0):.1f} casos")
        st.caption(f"Baseline sazonal: MAE {mbase.get('MAE', float('nan')):.1f} | RMSE {mbase.get('RMSE', float('nan')):.1f}. Avaliação cronológica, não erro in-sample.")
        fig_test = go.Figure()
        x_test = [f"{int(a)}-S{int(b):02d}" for a,b in zip(df_avaliacao.ano, df_avaliacao.semana_epidem)]
        fig_test.add_trace(go.Scatter(x=x_test, y=df_avaliacao.casos, mode='lines+markers', name='Observado', line=dict(color='#38BDF8')))
        fig_test.add_trace(go.Scatter(x=x_test, y=df_avaliacao.previsto_rf, mode='lines+markers', name='Previsto no teste futuro', line=dict(color='#F43F5E')))
        fig_test.update_layout(title='Teste cronológico: observado vs previsto', height=330, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#F8FAFC'))
        st.plotly_chart(fig_test, use_container_width=True)

    X = df_historico[features_cols]
    y_true = df_historico['casos']
    y_pred = modelo_rf.predict(X)
    st.caption('Os gráficos abaixo usam o histórico completo com o modelo refitado: diagnóstico exploratório in-sample, NÃO é prova de desempenho futuro.')

    st.markdown("---")
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### Explicabilidade: Importância das Variáveis")
        importances = modelo_rf.feature_importances_
        df_imp = pd.DataFrame({'Variável': features_cols, 'Importância': importances}).sort_values('Importância', ascending=True)
        
        fig_imp = px.bar(df_imp, x='Importância', y='Variável', orientation='h', color_discrete_sequence=['#38BDF8'])
        fig_imp.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#F8FAFC'))
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption("Importância por impureza do Random Forest. Não implica causalidade nem representa, necessariamente, a influência de cada variável em previsões futuras.")
        
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
    * **Justificativa:** O modelo capta relações não lineares entre as variáveis disponíveis, sem demonstrar relações causais.
    * **Validação:** A versão revisada usa holdout temporal (últimas semanas) para testar o modelo fora da amostra, comparando-o com uma referência sazonal. Consulte a aba Diagnóstico e os artefatos de avaliação.

    ### ⚠️ Confiabilidade e Limitações
    A transparência é um pilar do DengueRadar. É fundamental compreender as ressalvas da plataforma:
    1. **Atraso de Notificação:** Os dados observados sofrem atrasos institucionais. O modelo trata dados históricos como consolidados, o que pode não refletir a realidade em tempo real.
    2. **Distribuição Espacial Heurística:** A aba "Inteligência Territorial" distribui um total inserido pelo usuário usando pesos heurísticos fixos; as populações fornecidas são valores de referência aproximados e devem ser verificadas em fonte oficial. Os pontos representam áreas de **vulnerabilidade teórica estimada**, e não localizações exatas de pacientes.
    3. **Extrapolação Climática:** Simulações com parâmetros climáticos muito além do histórico de treinamento podem gerar previsões estatisticamente não confiáveis.
    4. **Sistema de Apoio:** A IA não toma decisões clínicas. As projeções apoiam o planejamento, mas as ações devem seguir os protocolos oficiais da Vigilância em Saúde.
""")
    
    st.markdown("---")
    st.info("Desenvolvido por alunos da FIAP - Matheus Alexandre e Sabrina Fraguela para integrar inovação social, tecnologia preditiva e inteligência cidadã.")