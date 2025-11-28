import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import yfinance as yf
import mplfinance as mpf
from pytickersymbols import PyTickerSymbols
import pandas as pd
from datetime import datetime, timedelta
import time
import io

# Define a arquitetura do modelo (dual-output matching stock_cnn.py)
def create_dual_output_model(num_cdl_patterns=20, dropout_rate=0.35):
    """
    Cria um modelo de saída dupla para análise de padrões de candlestick.
    
    Args:
        num_cdl_patterns: Número de padrões CDL (padrão: 20).
        dropout_rate: Taxa de dropout aplicada a todas as camadas (padrão: 0.35).
    
    Returns:
        Modelo Keras com duas saídas: 'cdl_patterns' e 'price_directions'
    """
    inputs = tf.keras.layers.Input(shape=(128, 128, 3), name='input_image')
    
    # Bloco 1: 32 filtros
    c1 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    c1 = tf.keras.layers.BatchNormalization()(c1)
    c1 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(c1)
    s2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(c1)
    s2 = tf.keras.layers.Dropout(dropout_rate)(s2)
    
    # Bloco 2: 64 filtros
    c3 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(s2)
    c3 = tf.keras.layers.BatchNormalization()(c3)
    c3 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(c3)
    s4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(c3)
    s4 = tf.keras.layers.Dropout(dropout_rate)(s4)
    
    # Bloco 3: 128 filtros
    c5 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(s4)
    c5 = tf.keras.layers.BatchNormalization()(c5)
    c5 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(c5)
    s6 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(c5)
    s6 = tf.keras.layers.Dropout(dropout_rate)(s6)
    
    # Achata características da CNN
    flat = tf.keras.layers.Flatten()(s6)
    
    # Camadas densas para processamento de características
    f7 = tf.keras.layers.Dense(256, activation='relu')(flat)
    f7 = tf.keras.layers.BatchNormalization()(f7)
    f7 = tf.keras.layers.Dropout(dropout_rate)(f7)
    f8 = tf.keras.layers.Dense(128, activation='relu')(f7)
    f8 = tf.keras.layers.BatchNormalization()(f8)
    f8 = tf.keras.layers.Dropout(dropout_rate)(f8)
    
    # Saída 1: Predições de Padrões CDL (classificação multi-label)
    cdl_patterns = tf.keras.layers.Dense(
        num_cdl_patterns,
        activation='sigmoid',
        name='cdl_patterns'
    )(f8)
    
    # Cabeça MLP: Predições de Direção de Preço
    mlp_hidden = tf.keras.layers.Dense(128, activation='relu')(cdl_patterns)
    mlp_hidden = tf.keras.layers.BatchNormalization()(mlp_hidden)
    mlp_hidden = tf.keras.layers.Dropout(dropout_rate)(mlp_hidden)
    
    # Saída 2: Predições de Direção de Preço (6 saídas binárias)
    # [next1_up, next1_down, next5_up, next5_down, next30_up, next30_down]
    price_directions = tf.keras.layers.Dense(
        6,
        activation='sigmoid',
        name='price_directions'
    )(mlp_hidden)
    
    return tf.keras.models.Model(
        inputs=inputs,
        outputs=[cdl_patterns, price_directions],
        name='cnn_mlp_dual_output'
    )

# Helper functions for live stock tracking
def search_stocks(query):
    """Search for stocks matching the query."""
    stock_data = PyTickerSymbols()
    us_indexes = ['DOW JONES', 'NASDAQ 100', 'S&P 100', 'S&P 500']
    
    all_stocks = []
    for index_name in us_indexes:
        try:
            stocks = stock_data.get_stocks_by_index(index_name)
            all_stocks.extend(stocks)
        except Exception:
            pass
    
    # Filter stocks matching query
    query_lower = query.lower()
    matches = []
    seen_symbols = set()
    
    for stock in all_stocks:
        symbol = stock['symbol']
        name = stock.get('name', '')
        
        # Avoid duplicates
        if symbol in seen_symbols:
            continue
        
        if query_lower in symbol.lower() or query_lower in name.lower():
            matches.append({
                'symbol': symbol,
                'name': name,
                'country': stock.get('country', 'US')
            })
            seen_symbols.add(symbol)
    
    return matches

def fetch_stock_data(ticker, period='1d', interval='1m'):
    """Fetch stock data from yfinance."""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        
        if data.empty:
            return None
        
        # Get last 30 rows
        data = data.tail(30)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def plot_candlestick_chart(data, ticker):
    """Plot candlestick chart using mplfinance."""
    try:
        # Create the plot
        fig, axes = mpf.plot(
            data,
            type='candle',
            style='yahoo',
            volume=True,
            figratio=(16, 9),
            figsize=(12, 7),
            returnfig=True,
            title=f"{ticker} - Last 30 Minutes"
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        return None

# Configuração da página
st.set_page_config(
    page_title="Previsão de Tendência de Ações",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("📱 Navegação")
page = st.sidebar.radio(
    "Escolha uma página:",
    ["🏠 Homepage", "📊 Live Stock Tracker"]
)

# ============================================================================
# PAGE 1: Homepage (Image Upload & Prediction)
# ============================================================================
if page == "🏠 Homepage":
    st.title("📈 Previsão de Tendência de Ações a partir de Imagens de Gráficos")
    
    st.markdown("""
    Bem-vindo! Este aplicativo utiliza um modelo de deep learning para prever se o preço de uma ação irá **subir** ou **cair** com base em uma imagem de gráfico enviada.

    **Importante:**  
    - As previsões são baseadas em análise de padrões candlestick e direções futuras de preço.
    - O modelo foi treinado exclusivamente para gráficos do tipo **candlestick**. O uso de outros tipos de gráficos ou imagens pode gerar resultados inesperados.

    **Como funciona:**  
    1. Faça o upload de uma imagem de gráfico candlestick (JPG, PNG ou JPEG).  
    2. A imagem será redimensionada e processada pelo modelo.  
    3. Você verá a previsão para diferentes horizontes de tempo (próximo 1, 5 e 30 períodos).
    """, unsafe_allow_html=True)
    
    st.markdown("### Passo 1: Faça o Upload de um Gráfico Candlestick")
    uploaded_file = st.file_uploader(
        "Formatos suportados: JPG, PNG e JPEG.",
        type=["jpg", "png", "jpeg"]
    )
    
    # Carrega o modelo uma vez
    @st.cache_resource
    def load_model():
        model = create_dual_output_model()
        model.load_weights("best_model.weights.h5")
        return model
    
    model = load_model()
    
    if uploaded_file is not None:
        st.markdown("### Passo 2: Pré-visualização e Processamento da Imagem")
        st.caption("""
        Tratamentos aplicados à imagem:
        - Conversão para o formato RGB (cores padrão).
        - Redimensionamento para 128x128 pixels para compatibilidade com o modelo.
        - Normalização dos valores dos pixels (de 0 a 1).
        Esses passos garantem que a imagem esteja no formato ideal para análise pelo modelo de rede neural.
        """)
        original_image = Image.open(uploaded_file).convert("RGB")
        resized_image = original_image.resize((128, 128))
        img_array = np.array(resized_image) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        
        tab1, tab2 = st.tabs(["🖼️ Imagem Original", "🔍 Imagem Redimensionada (128x128)"])
        with tab1:
            st.image(original_image, caption="Imagem original", use_container_width=True)
        with tab2:
            st.image(resized_image, caption="Redimensionada (128x128)", use_container_width=True)
        
        st.markdown("### Passo 3: Resultado da Previsão")
        st.caption("Esta previsão é baseada apenas na imagem de gráfico candlestick enviada e não constitui recomendação financeira.")
        
        with st.spinner("Analisando sua imagem..."):
            try:
                # Model returns [cdl_patterns, price_directions]
                cdl_preds, price_preds = model.predict(img_batch, verbose=0)
                
                # Price predictions: [next1_up, next1_down, next5_up, next5_down, next30_up, next30_down]
                price_directions = price_preds[0]
                
                # Extract predictions for each horizon
                next1_up = float(price_directions[0])
                next1_down = float(price_directions[1])
                next5_up = float(price_directions[2])
                next5_down = float(price_directions[3])
                next30_up = float(price_directions[4])
                next30_down = float(price_directions[5])
                
                # Display predictions for each time horizon
                st.markdown("#### 📊 Previsões de Direção de Preço")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Próximo 1 período (t+1)**")
                    if next1_up > next1_down:
                        st.success(f"📈 Subir ({next1_up:.1%})")
                    else:
                        st.warning(f"📉 Cair ({next1_down:.1%})")
                
                with col2:
                    st.markdown("**Próximos 5 períodos (t+5)**")
                    if next5_up > next5_down:
                        st.success(f"📈 Subir ({next5_up:.1%})")
                    else:
                        st.warning(f"📉 Cair ({next5_down:.1%})")
                
                with col3:
                    st.markdown("**Próximos 30 períodos (t+30)**")
                    if next30_up > next30_down:
                        st.success(f"📈 Subir ({next30_up:.1%})")
                    else:
                        st.warning(f"📉 Cair ({next30_down:.1%})")
                
                # Detailed probabilities
                with st.expander("📈 Ver Probabilidades Detalhadas"):
                    st.markdown("**Próximo 1 período (t+1):**")
                    st.progress(next1_up, text=f"Subir: {next1_up:.2%}")
                    st.progress(next1_down, text=f"Cair: {next1_down:.2%}")
                    
                    st.markdown("**Próximos 5 períodos (t+5):**")
                    st.progress(next5_up, text=f"Subir: {next5_up:.2%}")
                    st.progress(next5_down, text=f"Cair: {next5_down:.2%}")
                    
                    st.markdown("**Próximos 30 períodos (t+30):**")
                    st.progress(next30_up, text=f"Subir: {next30_up:.2%}")
                    st.progress(next30_down, text=f"Cair: {next30_down:.2%}")
                
                st.caption("**Nota:** Esta previsão é baseada apenas na imagem de gráfico candlestick enviada e não constitui recomendação financeira.")
                
            except Exception as e:
                st.error(f"Não foi possível carregar o modelo ou realizar a previsão: {e}")
                st.error("Verifique se o arquivo 'best_model.weights.h5' é compatível com a arquitetura dual-output.")
            
            st.markdown("---")
            if "show_dialog" not in st.session_state:
                st.session_state.show_dialog = False
            
            def feedback_dialog():
                st.markdown("Ajude-nos a melhorar! Preencha as informações abaixo:")
                
                ticker = st.text_input("Ticker do ativo (*)", placeholder="Ex: PETR4")
                col1, col2 = st.columns(2)
                with col1:
                    data_inicio = st.date_input("Data inicial do gráfico (*)")
                    hora_inicio = st.time_input("Hora inicial do gráfico (*)")
                with col2:
                    data_fim = st.date_input("Data final do gráfico (*)")
                    hora_fim = st.time_input("Hora final do gráfico (*)")
                url_fonte = st.text_input("Fonte dos dados (URL) (opcional)", placeholder="Cole aqui o link da fonte")
                acerto = st.radio("O modelo acertou a previsão? (*)", ["Sim", "Não"])
                email = st.text_input("Seu e-mail (opcional)", placeholder="Para receber novidades do projeto")
                
                obrigatorios_preenchidos = (
                    ticker.strip() != "" and
                    data_inicio is not None and
                    hora_inicio is not None and
                    data_fim is not None and
                    hora_fim is not None and
                    acerto in ["Sim", "Não"]
                )
                
                if not obrigatorios_preenchidos:
                    st.warning("Por favor, preencha todos os campos obrigatórios marcados com (*).")
                
                enviar = st.button("Enviar Feedback", key="enviar_feedback", disabled=not obrigatorios_preenchidos)
                
                if enviar:
                    feedback_obj = {
                        "ticker": ticker,
                        "data_inicio": str(data_inicio),
                        "hora_inicio": str(hora_inicio),
                        "data_fim": str(data_fim),
                        "hora_fim": str(hora_fim),
                        "url_fonte": url_fonte,
                        "acerto": acerto,
                        "email": email
                    }
                    st.success("Obrigado pelo seu feedback! Sua resposta foi registrada com sucesso. 😊")
                    st.json(feedback_obj)
                    st.session_state.show_dialog = False
            
            if st.button("Abrir Formulário de Feedback"):
                st.session_state.show_dialog = True
            
            if st.session_state.show_dialog:
                st.dialog("Formulário de Feedback")(feedback_dialog)()
    else:
        st.info("Por favor, faça o upload de uma imagem de gráfico candlestick para começar.")

# ============================================================================
# PAGE 2: Live Stock Tracker
# ============================================================================
elif page == "📊 Live Stock Tracker":
    st.title("📊 Rastreador de Ações ao Vivo")
    
    st.markdown("""
    Acompanhe os últimos **30 minutos** de negociação de qualquer ação listada nos principais índices dos EUA.
    O gráfico é atualizado automaticamente a cada minuto com os dados mais recentes.
    """)
    
    # Initialize session state
    if 'selected_ticker' not in st.session_state:
        st.session_state.selected_ticker = None
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    # Search section
    st.markdown("### 🔍 Buscar Ação")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Digite o símbolo ou nome da ação:",
            placeholder="Ex: AAPL, Tesla, Microsoft"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.button("🔍 Buscar", use_container_width=True)
    
    # Handle search
    if search_button and search_query:
        with st.spinner("Buscando ações..."):
            matches = search_stocks(search_query)
            
            if len(matches) == 0:
                st.warning("Nenhuma ação encontrada com esse critério de busca.")
            elif len(matches) == 1:
                # Auto-select single match
                st.session_state.selected_ticker = matches[0]['symbol']
                st.success(f"✅ Ação selecionada: **{matches[0]['symbol']}** - {matches[0]['name']}")
            else:
                # Show dialog with multiple matches
                @st.dialog("Selecione uma ação")
                def select_stock():
                    st.markdown(f"Encontradas **{len(matches)}** ações correspondentes:")
                    
                    for i, match in enumerate(matches[:10]):  # Limit to 10 results
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.markdown(f"**{match['symbol']}** - {match['name']}")
                        with col_b:
                            if st.button("Selecionar", key=f"select_{i}"):
                                st.session_state.selected_ticker = match['symbol']
                                st.rerun()
                    
                    if len(matches) > 10:
                        st.info(f"Mostrando 10 de {len(matches)} resultados. Refine sua busca para ver mais opções.")
                
                select_stock()
    
    # Display selected ticker and chart
    if st.session_state.selected_ticker:
        st.markdown("---")
        st.markdown(f"### 📈 Gráfico ao Vivo: **{st.session_state.selected_ticker}**")
        
        # Control buttons
        col1, col2, col3 = st.columns([2, 2, 8])
        
        with col1:
            if st.button("🔄 Atualizar Agora"):
                st.session_state.last_refresh = time.time()
                st.rerun()
        
        with col2:
            if st.session_state.auto_refresh:
                if st.button("⏸️ Pausar Auto-Refresh"):
                    st.session_state.auto_refresh = False
                    st.rerun()
            else:
                if st.button("▶️ Ativar Auto-Refresh"):
                    st.session_state.auto_refresh = True
                    st.session_state.last_refresh = time.time()
                    st.rerun()
        
        # Fetch and display data
        with st.spinner("Carregando dados..."):
            data = fetch_stock_data(st.session_state.selected_ticker)
            
            if data is not None and not data.empty:
                # Display chart
                fig = plot_candlestick_chart(data, st.session_state.selected_ticker)
                if fig:
                    st.pyplot(fig)
                
                # Show last update time and metrics
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    last_close = data['Close'].iloc[-1]
                    st.metric("Último Preço", f"${last_close:.2f}")
                
                with col_b:
                    change = data['Close'].iloc[-1] - data['Close'].iloc[0]
                    change_pct = (change / data['Close'].iloc[0]) * 100
                    st.metric("Variação (30min)", f"${change:.2f}", f"{change_pct:+.2f}%")
                
                with col_c:
                    volume = data['Volume'].sum()
                    st.metric("Volume Total", f"{volume:,.0f}")
                
                with col_d:
                    update_time = datetime.now().strftime("%H:%M:%S")
                    st.metric("Última Atualização", update_time)
                
                # Auto-refresh logic
                if st.session_state.auto_refresh:
                    elapsed = time.time() - st.session_state.last_refresh
                    remaining = max(0, 60 - int(elapsed))
                    
                    if remaining > 0:
                        st.info(f"⏱️ Próxima atualização em {remaining} segundos...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.session_state.last_refresh = time.time()
                        st.rerun()
            else:
                st.error("⚠️ Não foi possível carregar os dados desta ação. Verifique se o mercado está aberto ou tente outra ação.")
    else:
        st.info("👆 Use a busca acima para selecionar uma ação e visualizar o gráfico ao vivo.")
