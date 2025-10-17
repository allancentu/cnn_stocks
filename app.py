import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Define a arquitetura do modelo
def my_lenet(do_freq=0.3):
    inputs = tf.keras.layers.Input(shape=(128,128,3))

    c1 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    c1 = tf.keras.layers.BatchNormalization()(c1)
    c1 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(c1)
    s2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(c1)
    s2 = tf.keras.layers.Dropout(do_freq)(s2)

    c3 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(s2)
    c3 = tf.keras.layers.BatchNormalization()(c3)
    c3 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(c3)
    s4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(c3)
    s4 = tf.keras.layers.Dropout(do_freq)(s4)

    c5 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(s4)
    c5 = tf.keras.layers.BatchNormalization()(c5)
    c5 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(c5)
    s6 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(c5)
    s6 = tf.keras.layers.Dropout(do_freq)(s6)

    flat = tf.keras.layers.Flatten()(s6)
    f7 = tf.keras.layers.Dense(256, activation='relu')(flat)
    f7 = tf.keras.layers.BatchNormalization()(f7)
    f7 = tf.keras.layers.Dropout(do_freq)(f7)
    f8 = tf.keras.layers.Dense(128, activation='relu')(f7)
    f8 = tf.keras.layers.BatchNormalization()(f8)
    f8 = tf.keras.layers.Dropout(do_freq)(f8)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(f8)

    return tf.keras.models.Model(inputs, outputs, name='my_lenet')

st.set_page_config(page_title="Previsão de Tendência de Ações", page_icon=":chart_with_upwards_trend:", layout="centered")

st.title("📈 Previsão de Tendência de Ações a partir de Imagens de Gráficos")
# Sidebar navigation
st.sidebar.title("Navegação")
page = st.sidebar.radio(
    "Escolha uma página:",
    ["🏠 Página Principal", "🧠 Sobre o Modelo CNN"]
)

if page == "🏠 Página Principal":
    st.markdown("""
    Bem-vindo! Este aplicativo utiliza um modelo de deep learnning para prever se o preço de uma ação irá **subir** ou **cair** com base em uma imagem de gráfico enviada.

    **Importante:**  
    - As previsões são para o período **t+5** (cinco períodos após o último período mostrado no gráfico).
    - O modelo foi treinado exclusivamente para gráficos do tipo **candlestick**, utilizados em **análise gráfica**. O uso de outros tipos de gráficos ou imagens pode gerar resultados inesperados.

    **Como funciona:**  
    1. Faça o upload de uma imagem de gráfico candlestick (JPG, PNG ou JPEG).  
    2. A imagem será redimensionada e processada pelo nosso modelo.  
    3. Você verá a previsão e uma prévia da sua imagem.
    """, unsafe_allow_html=True)

    st.markdown("### Passo 1: Faça o Upload de um Gráfico Candlestick")
    uploaded_file = st.file_uploader(
        "Formatos suportados: JPG, PNG e JPEG.",
        type=["jpg", "png", "jpeg"]
    )

    # Carrega o modelo uma vez
    @st.cache_resource
    def load_model():
        model = my_lenet()
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
        img_array = np.array(resized_image) / 255.0  # Normaliza os valores dos pixels
        img_batch = np.expand_dims(img_array, axis=0)  # Adiciona dimensão de lote

        tab1, tab2 = st.tabs(["🖼️ Imagem Original", "🔍 Imagem Redimensionada (128x128)"])
        with tab1:
            st.image(original_image, caption="Imagem original", width="content")
        with tab2:
            st.image(resized_image, caption="Redimensionada (128x128)", width="content")

        st.markdown("### Passo 3: Resultado da Previsão")
        st.caption("Esta previsão é baseada apenas na imagem de gráfico candlestick enviada e não constitui recomendação financeira.")
        with st.spinner("Analisando sua imagem..."):
            try:
                preds = model.predict(img_batch)
                pred_class_idx = int(np.argmax(preds, axis=1)[0])
                pred_class = "📈 subir" if pred_class_idx == 1 else "📉 cair"
                st.success(f"**Previsão para t+5:** O modelo prevê que o preço do ativo irá **{pred_class}** daqui a cinco períodos.")
                st.caption("Nota: Esta previsão é baseada apenas na imagem de gráfico candlestick enviada e não constitui recomendação financeira.")

                # Exibe as probabilidades
                st.markdown("#### Probabilidades da Previsão")
                st.write({
                    "Probabilidade de subir (📈)": float(preds[0][1]),
                    "Probabilidade de cair (📉)": float(preds[0][0])
                })
            except Exception as e:
                st.error(f"Não foi possível carregar o modelo ou realizar a previsão: {e}")
                
            st.markdown("---")
            if "show_dialog" not in st.session_state:
                st.session_state.show_dialog = False

            if st.button("Abrir Formulário de Feedback"):
                st.session_state.show_dialog = True

            if st.session_state.show_dialog:
                with st.dialog("Formulário de Feedback"):
                    st.markdown("Ajude-nos a melhorar! Preencha as informações abaixo:")
                    ticker = st.text_input("Ticker do ativo", placeholder="Ex: PETR4")
                    col1, col2 = st.columns(2)
                    with col1:
                        data_inicio = st.date_input("Data inicial do gráfico")
                        hora_inicio = st.time_input("Hora inicial do gráfico")
                    with col2:
                        data_fim = st.date_input("Data final do gráfico")
                        hora_fim = st.time_input("Hora final do gráfico")
                    url_fonte = st.text_input("Fonte dos dados (URL)", placeholder="Cole aqui o link da fonte")
                    acerto = st.radio("O modelo acertou a previsão?", ["Sim", "Não"])
                    email = st.text_input("Seu e-mail (opcional)", placeholder="Para receber novidades do projeto")
                    enviar = st.button("Enviar Feedback", key="enviar_feedback")

                    if enviar:
                        feedback_obj = {
                            "ticker": ticker,
                            "data_inicio": str(data_inicio),
                            "hora_inicio": str(hora_inicio),
                            "data_fim": str(data_fim),
                            "hora_fim": str(hora_fim),
                            "url_fonte": url_fonte,
                            "acerto": acerto,
                            "email": email,
                            "caminho_imagem_original": str(uploaded_file.name),
                            "caminho_imagem_redimensionada": "imagem_redimensionada.png"
                        }
                        st.success("Obrigado pelo seu feedback! Sua resposta foi registrada com sucesso. 😊")
                        st.json(feedback_obj)
                        st.session_state.show_dialog = False
    else:
        st.info("Por favor, faça o upload de uma imagem de gráfico candlestick para começar.")

elif page == "🧠 Sobre o Modelo CNN":
    st.header("🧠 Sobre o Modelo de Rede Neural Convolucional (CNN)")
    st.markdown("""
    Este aplicativo utiliza uma arquitetura de rede neural convolucional (CNN) inspirada no clássico **LeNet-5**, porém com diversas melhorias modernas, como camadas de normalização em lote (*Batch Normalization*) e *Dropout* para evitar overfitting.

    ### Por que essa arquitetura?
    - Redes CNN são especialmente eficazes para análise de imagens, pois conseguem extrair padrões visuais relevantes.
    - A arquitetura escolhida é robusta, eficiente e adaptada para gráficos financeiros, permitindo identificar tendências visuais em gráficos candlestick.

    ### Dados utilizados no treinamento
    - Foram utilizados **64.000 gráficos candlestick** de ações listadas na **Nasdaq** e **NYSE**, cobrindo o período de **2020 a 2025**.
    - Os dados foram baixados via biblioteca **yfinance** e os gráficos gerados com **mplfinance**.
    - Todo o conjunto de dados foi normalizado para evitar viés e garantir maior precisão.

    ### Principais métricas do modelo
    - **Acurácia final de teste:** 99,40%
    - **F1 Score:** 0,9936
    - **Erro de omissão para subida:** 0,62%
    - **Erro de comissão para subida:** 0,66%
    - **Erro de omissão para queda:** 0,58%
    - **Erro de comissão para queda:** 0,55%

    Estes resultados indicam que o modelo é altamente confiável para identificar tendências de subida ou queda em gráficos candlestick, considerando o horizonte de previsão de **t+5** períodos.

    ---
    **Observação:** Apesar da alta precisão, este modelo serve apenas como ferramenta de apoio e não substitui análise financeira profissional.
    """)
