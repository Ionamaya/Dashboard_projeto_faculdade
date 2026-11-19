import os
import re
import unicodedata
from io import BytesIO as by

import numpy as num
import pandas as pa
import plotly.express as px
import plotly.graph_objects as go
import polars as po
import streamlit as st

# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

# Coloque o arquivo da logo na MESMA PASTA deste dashboard.py,
# com um destes nomes: logo.png, logo.jpg, logo.jpeg ou logo.webp
NOMES_LOGO = ["logo.png", "logo.jpg", "logo.jpeg", "logo.webp"]


def _versao_streamlit():
    try:
        return tuple(int(p) for p in st.__version__.split(".")[:2])
    except Exception:
        return (0, 0)


# Versões novas do Streamlit trocaram use_container_width=True por width="stretch".
if _versao_streamlit() >= (1, 50):
    LARGURA_TOTAL = {"width": "stretch"}
else:
    LARGURA_TOTAL = {"use_container_width": True}


# ============================================================
# FUNÇÕES DE AUXÍLIO
# ============================================================

def encontrar_logo():
    """Procura o arquivo da logo ao lado deste script. Retorna o caminho ou None."""
    pasta = os.path.dirname(os.path.abspath(__file__))
    for nome in NOMES_LOGO:
        caminho = os.path.join(pasta, nome)
        if os.path.exists(caminho):
            return caminho
    return None


def coverter_numer(serie):
    """
    Converte uma série para números.
    Valores que não forem numéricos são ignorados.
    """
    valores = pa.to_numeric(serie, errors="coerce")
    return valores.dropna()


def normalizar_texto(texto):
    """Padroniza texto para comparar respostas equivalentes."""
    if pa.isna(texto):
        return texto
    texto = str(texto).strip().casefold()
    texto = re.sub(r"\s+", " ", texto)
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in texto if not unicodedata.combining(c))


def criar_mapa_normalizacao(texto_mapeamento):
    """Lê regras no formato variacao=forma_padrao."""
    mapa = {}
    for linha in texto_mapeamento.splitlines():
        if "=" in linha:
            origem, destino = linha.split("=", 1)
            origem, destino = normalizar_texto(origem), normalizar_texto(destino)
            if origem and destino:
                mapa[origem] = destino
    return mapa


def aplicar_normalizacao(serie, mapa, ativar=True):
    """Normaliza texto sem modificar respostas numéricas."""
    if not ativar:
        return serie

    def ajustar(valor):
        if not isinstance(valor, str):
            return valor
        texto = normalizar_texto(valor)
        if texto in mapa:
            return mapa[texto]
        for origem in sorted(mapa, key=len, reverse=True):
            texto = re.sub(r"(?<!\w)" + re.escape(origem) + r"(?!\w)", mapa[origem], texto)
        return texto

    return serie.map(ajustar)


def calcular_numero_classes(valores):
    """Número de classes pela regra de Sturges (entre 2 e 20)."""
    if len(valores) >= 2:
        return max(2, min(20, int(num.ceil(1 + 3.322 * num.log10(len(valores))))))
    return 2


# ------------------------------------------------------------
# TABELAS DE FREQUÊNCIA
# ------------------------------------------------------------

def criar_tabela_resposta(serie_pa):
    """
    Tabela de frequência por resposta, com:
    - frequência absoluta
    - frequência relativa (%)
    - frequência acumulada
    - frequência acumulada (%)

    Se todas as respostas forem números, a tabela fica em ordem crescente
    de valor. Caso contrário, fica da resposta mais frequente para a menos
    frequente. A acumulada segue essa ordem.
    """
    contagem = serie_pa.dropna().value_counts()
    contagem = contagem[contagem.index.astype(str).str.strip() != ""]

    chaves = pa.to_numeric(pa.Series(contagem.index), errors="coerce").to_numpy(dtype=float)
    if len(contagem) > 0 and not num.isnan(chaves).any():
        contagem = contagem.iloc[num.argsort(chaves, kind="stable")]
    else:
        contagem = contagem.sort_values(ascending=False, kind="stable")

    tabela = pa.DataFrame({
        "Resposta": contagem.index.astype(str),
        "Frequência absoluta": contagem.values,
    })
    total = tabela["Frequência absoluta"].sum()
    if total > 0:
        tabela["Frequência relativa (%)"] = tabela["Frequência absoluta"] / total * 100
        tabela["Frequência acumulada"] = tabela["Frequência absoluta"].cumsum()
        tabela["Frequência acumulada (%)"] = tabela["Frequência acumulada"] / total * 100
    else:
        tabela["Frequência relativa (%)"] = []
        tabela["Frequência acumulada"] = []
        tabela["Frequência acumulada (%)"] = []
    return tabela


def intervalo_classe(valores, numero_de_classe):
    """Cria os limites dos intervalos de classe."""
    mini = valores.min()
    maxi = valores.max()

    if mini == maxi:
        return num.array([mini - 0.5, maxi + 0.5])

    return num.linspace(mini, maxi, numero_de_classe + 1)


def criar_tabela_intervalo(valores, numero_de_classe):
    """Cria uma tabela de frequência com intervalos de classe."""
    intervalos = intervalo_classe(valores, numero_de_classe)

    if intervalos is None:
        return None

    classe = pa.cut(valores, bins=intervalos, include_lowest=True)
    frequencias = classe.value_counts(sort=False)

    tabela = pa.DataFrame({
        "Intervalo": frequencias.index.astype(str),
        "Frequência": frequencias.values,
    })
    tabela["Ponto médio"] = (intervalos[:-1] + intervalos[1:]) / 2

    total = tabela["Frequência"].sum()
    tabela["Frequência relativa (%)"] = tabela["Frequência"] / total * 100
    tabela["Frequência acumulada"] = tabela["Frequência"].cumsum()
    tabela["Frequência acumulada (%)"] = tabela["Frequência acumulada"] / total * 100

    return tabela, classe, intervalos


# ------------------------------------------------------------
# MEDIDAS PARA DADOS AGRUPADOS EM INTERVALOS
# ------------------------------------------------------------

def calcular_media_intervalo(valores, numero_de_classe):
    """Média aproximada utilizando os pontos médios dos intervalos."""
    resultado = criar_tabela_intervalo(valores, numero_de_classe)
    if resultado is None:
        return None

    tabela, classe, intervalos = resultado
    pontos_medios = (intervalos[:-1] + intervalos[1:]) / 2
    frequencias = tabela["Frequência"].values

    return (pontos_medios * frequencias).sum() / frequencias.sum()


def _posicao_no_intervalo(valores, numero_de_classe, fracao):
    """
    Fórmula de interpolação usada pela mediana e pelos quartis
    em dados agrupados. fracao = 0.25 (Q1), 0.5 (mediana/Q2), 0.75 (Q3).
    """
    resultado = criar_tabela_intervalo(valores, numero_de_classe)
    if resultado is None:
        return None

    tabela, classe, intervalos = resultado
    frequencias = tabela["Frequência"].values
    posicao = frequencias.sum() * fracao
    acumuladas = num.cumsum(frequencias)

    indice = num.where(acumuladas >= posicao)[0][0]
    frequencia_classe = frequencias[indice]
    acumulada_anterior = acumuladas[indice - 1] if indice > 0 else 0
    limite_inferior = intervalos[indice]
    amplitude = intervalos[indice + 1] - intervalos[indice]

    return limite_inferior + ((posicao - acumulada_anterior) / frequencia_classe) * amplitude


def calcular_mediana_intervalo(valores, numero_de_classe):
    """Mediana aproximada para dados agrupados em intervalos."""
    return _posicao_no_intervalo(valores, numero_de_classe, 0.5)


def quartil_intervalo(valores, numero_de_classe, quartil):
    """Q1, Q2 ou Q3 para dados agrupados em intervalos."""
    fracao = {"Q1": 0.25, "Q2": 0.5, "Q3": 0.75}.get(quartil, 0.75)
    return _posicao_no_intervalo(valores, numero_de_classe, fracao)


def calcular_moda_intervalada(valores, numero_de_classe):
    """Moda aproximada (Czuber) para dados agrupados em intervalos."""
    resultado = criar_tabela_intervalo(valores, numero_de_classe)
    if resultado is None:
        return None

    tabela, classe, intervalos = resultado
    frequencias = tabela["Frequência"].values

    indice = num.argmax(frequencias)
    frequencia_modal = frequencias[indice]
    frequencia_anterior = frequencias[indice - 1] if indice > 0 else 0
    frequencia_posterior = frequencias[indice + 1] if indice < len(frequencias) - 1 else 0

    limite_inferior = intervalos[indice]
    amplitude = intervalos[indice + 1] - intervalos[indice]

    diferenca_anterior = frequencia_modal - frequencia_anterior
    diferenca_posterior = frequencia_modal - frequencia_posterior
    denominador = diferenca_anterior + diferenca_posterior

    if denominador == 0:
        return intervalos[indice] + amplitude / 2

    return limite_inferior + (diferenca_anterior / denominador) * amplitude


# ------------------------------------------------------------
# GRÁFICOS
# ------------------------------------------------------------

def figura_png(fig):
    """Converte um gráfico Plotly para PNG em memória."""
    imagem = by()
    imagem.write(fig.to_image(format="png", scale=2))
    imagem.seek(0)
    return imagem


def figura_barras_simples(tabela_grafico):
    """Gráfico de barras usado quando o gráfico escolhido não pode ser feito."""
    return px.bar(
        tabela_grafico,
        x="Frequência absoluta",
        y="Resposta",
        orientation="h",
        title="Frequência das respostas",
    )


def criar_figura(grafico, tabela_grafico, serie_pa, numero_classes):
    """Monta a figura Plotly de acordo com o tipo de gráfico escolhido."""

    if grafico == "Gráfico de barras":
        tabela_grafico = tabela_grafico.sort_values(by="Frequência absoluta", ascending=True)
        fig = px.bar(
            tabela_grafico,
            x="Frequência absoluta",
            y="Resposta",
            orientation="h",
            text="Frequência absoluta",
            title="Frequência das respostas",
            labels={"Frequência absoluta": "Frequência", "Resposta": "Resposta"},
            hover_data={"Frequência relativa (%)": ":.2f"},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            yaxis={"categoryorder": "array", "categoryarray": tabela_grafico["Resposta"].tolist()},
            height=max(450, len(tabela_grafico) * 38),
            margin=dict(l=20, r=45, t=65, b=40),
        )

    elif grafico == "Gráfico de pizza":
        fig = px.pie(
            tabela_grafico,
            names="Resposta",
            values="Frequência absoluta",
            title="Distribuição das respostas",
            hole=0.25,
            hover_data=["Frequência relativa (%)"],
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Quantidade: %{value}<br>"
                "Percentual: %{percent}<extra></extra>"
            ),
        )
        fig.update_layout(height=550, margin=dict(l=20, r=20, t=65, b=30))

    elif grafico == "Gráfico de pontos":
        fig = px.scatter(
            tabela_grafico,
            x="Resposta",
            y="Frequência absoluta",
            size="Frequência absoluta",
            hover_name="Resposta",
            hover_data={"Frequência relativa (%)": ":.2f"},
            title="Frequência por resposta",
            labels={"Resposta": "Resposta", "Frequência absoluta": "Frequência"},
        )
        fig.update_traces(marker=dict(opacity=0.8))
        fig.update_layout(height=500, xaxis_tickangle=-35)

    elif grafico == "Gráfico de linhas":
        fig = px.line(
            tabela_grafico,
            x="Resposta",
            y="Frequência absoluta",
            markers=True,
            title="Frequência das respostas",
            hover_data={"Frequência relativa (%)": ":.2f"},
            labels={"Resposta": "Resposta", "Frequência absoluta": "Frequência"},
        )
        fig.update_traces(line=dict(width=3), marker=dict(size=9))
        fig.update_layout(height=500, xaxis_tickangle=-35)

    elif grafico == "Ogiva":
        valores_ogiva = coverter_numer(serie_pa)

        if len(valores_ogiva) < 2:
            st.warning("A ogiva precisa de pelo menos dois dados numéricos. A tabela e a medida estatística serão exibidas normalmente.")
            fig = figura_barras_simples(tabela_grafico)
        else:
            tabela_ogiva, classe_ogiva, intervalos_ogiva = criar_tabela_intervalo(valores_ogiva, numero_classes)
            frequencias_acumuladas = tabela_ogiva["Frequência"].cumsum()
            x_ogiva = num.insert(intervalos_ogiva[1:], 0, intervalos_ogiva[0])
            y_ogiva = num.insert(frequencias_acumuladas.to_numpy(), 0, 0)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_ogiva,
                y=y_ogiva,
                mode="lines+markers",
                name="Frequência acumulada",
                line=dict(width=3),
                marker=dict(size=9),
                hovertemplate="Limite: %{x:.2f}<br>Frequência acumulada: %{y}<extra></extra>",
            ))
            fig.update_layout(
                title="Ogiva — frequência acumulada",
                xaxis_title="Limite da classe",
                yaxis_title="Frequência acumulada",
                height=500,
                hovermode="x unified",
            )

    else:  # Boxplot
        valores_boxplot = coverter_numer(serie_pa)

        if len(valores_boxplot) < 1:
            st.warning("O boxplot precisa de dados numéricos. A tabela e a medida estatística serão exibidas normalmente.")
            fig = figura_barras_simples(tabela_grafico)
        else:
            fig = px.box(
                x=valores_boxplot,
                points="all",
                title="Distribuição dos valores",
                labels={"x": "Valores"},
            )
            fig.update_traces(boxmean=True, jitter=0.35, pointpos=0)
            fig.update_layout(height=500)

    # Estilo comum: zoom, seleção e dicas ao passar o mouse.
    fig.update_layout(
        template="plotly_white",
        hoverlabel=dict(namelength=-1),
        font=dict(size=13),
        legend_title_text="Respostas",
    )
    return fig


# ------------------------------------------------------------
# RESULTADO ESTATÍSTICO + TABELA DE FREQUÊNCIA
# ------------------------------------------------------------

def mostrar_resultado_e_tabela(medida, quartil, valores, numero_classes, tabela_frequencia):
    """
    Mostra, nesta ordem:
    1) a medida estatística escolhida e o seu resultado;
    2) a tabela de frequência (absoluta, relativa e acumulada).
    """
    tabela_exibicao = tabela_frequencia
    titulo_tabela = "### 📋 Tabela de frequência por resposta"

    st.markdown("### 📌 Resultado estatístico")

    if medida == "Moda":
        # A moda usa as respostas originais, então vale para texto e para número.
        maior_frequencia = tabela_frequencia["Frequência absoluta"].max()

        if maior_frequencia == 1:
            st.write("**Moda:** Não existe moda (nenhuma resposta se repete).")
        else:
            modas = tabela_frequencia.loc[
                tabela_frequencia["Frequência absoluta"] == maior_frequencia, "Resposta"
            ].tolist()
            st.write(f"**Moda:** {', '.join(modas)}")

    # Média, mediana e quartis precisam de dados numéricos.
    elif len(valores) == 0:
        st.warning("Esta medida precisa de respostas numéricas. Escolha Moda para analisar respostas textuais.")

    else:
        if medida == "Média":
            resultado_medida = calcular_media_intervalo(valores, numero_classes)
            nome_medida = "Média aproximada"
        elif medida == "Mediana":
            resultado_medida = calcular_mediana_intervalo(valores, numero_classes)
            nome_medida = "Mediana aproximada"
        else:
            resultado_medida = quartil_intervalo(valores, numero_classes, quartil)
            nome_medida = f"{quartil} aproximado"

        if resultado_medida is not None:
            st.write(f"**{nome_medida}:** {resultado_medida:.2f}")

        # A medida foi calculada pelos intervalos, então a tabela mostra os mesmos intervalos.
        tabela_classes = criar_tabela_intervalo(valores, numero_classes)[0]
        tabela_exibicao = tabela_classes.rename(columns={"Frequência": "Frequência absoluta"})
        titulo_tabela = "### 📋 Tabela de frequência por intervalos"

    st.markdown(titulo_tabela)
    st.dataframe(tabela_exibicao.round(2), hide_index=True, **LARGURA_TOTAL)
    st.caption(f"Total de respostas consideradas na tabela: {int(tabela_exibicao['Frequência absoluta'].sum())}")


# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Dashboard -Inotive",
    page_icon="🗃️",
    layout="wide",
)

# ============================================================
# CABEÇALHO COM A LOGO
# ============================================================

caminho_logo = encontrar_logo()

if caminho_logo:
    col_logo, col_titulo = st.columns([1, 4])
    with col_logo:
        st.image(caminho_logo, width=180)
    with col_titulo:
        st.title("📊 Dashboard Estatístico - Inotive")
else:
    st.title("📊 Dashboard Estatístico -Inotive")
    st.caption("ℹ️ Para exibir a logo, salve o arquivo como logo.png na mesma pasta deste dashboard.py.")

st.write(
    "Envie o arquivo Excel com as respostas "
    "do questionário para realizar as análises."
)

# ============================================================
# ENVIO DO ARQUIVO
# ============================================================

arquivo = st.file_uploader(
    "📁 Envie aqui o arquivo Excel escolhido para análise",
    type=["xls", "xlsx"],
)

# ============================================================
# LEITURA DO ARQUIVO
# ============================================================

if arquivo is not None:

    try:
        dados = po.read_excel(arquivo)
        st.success("✅ Arquivo carregado com sucesso!")
    except Exception as erro:
        st.error(f"❌ Não foi possível ler o arquivo: {erro}")
        st.stop()

    # ========================================================
    # DADOS
    # ========================================================

    st.subheader("📋 Dados")
    st.dataframe(dados, **LARGURA_TOTAL)
    st.write("Quantidade de linhas:", dados.height)
    st.write("Quantidade de colunas:", dados.width)

    # ========================================================
    # LISTA DE PERGUNTAS
    # ========================================================

    perguntas = [
        coluna
        for coluna in dados.columns
        if coluna.lower() not in ["carimbo de data/hora", "timestamp"]
    ]

    # ========================================================
    # PADRONIZAÇÃO DAS RESPOSTAS TEXTUAIS
    # ========================================================

    st.subheader("🧹 Padronização das respostas")
    st.write("O filtro reúne diferenças de maiúsculas, minúsculas, acentos e espaços. Você também pode cadastrar abreviações e diminutivos.")

    ativar_normalizacao = st.checkbox(
        "Ativar padronização automática",
        value=True,
        key="ativar_normalizacao",
    )

    with st.expander("Configurar abreviações e diminutivos"):
        st.caption("Uma regra por linha: variacao=forma_padrao. Exemplo: mercadinho=supermercado. As conversões são configuráveis para evitar unir respostas com significados diferentes.")
        texto_mapeamento = st.text_area(
            "Regras de padronização",
            value=(
                "AP. Goiania=aparecida de goiania\n"
                "AP Goiania=aparecida de goiania\n"
                "Aparecida de Goiânia=aparecida de goiania\n"
                "Aparecida de Goiania=aparecida de goiania\n"
                "aparecida de goiania=aparecida de goiania\n"
            ),
            height=150,
            key="regras_normalizacao_v2",
        )

    mapa_normalizacao = criar_mapa_normalizacao(texto_mapeamento)

    # ========================================================
    # ANÁLISE DAS PERGUNTAS
    # ========================================================

    st.header("📊 Análise das perguntas")

    for numero, pergunta in enumerate(perguntas, start=1):

        st.divider()
        st.subheader(f"{numero}. {pergunta}")

        # ----------------------------------------------------
        # MEDIDA ESTATÍSTICA
        # ----------------------------------------------------

        medida = st.selectbox(
            "📐 Medida estatística:",
            ["Moda", "Média", "Mediana", "Quartil"],
            key=f"medida_{numero}",
        )

        quartil = None
        if medida == "Quartil":
            quartil = st.selectbox(
                "Escolha o quartil:",
                ["Q1", "Q2", "Q3"],
                key=f"quartil_{numero}",
            )

        # ----------------------------------------------------
        # ESCOLHA DO GRÁFICO
        # ----------------------------------------------------

        grafico = st.selectbox(
            "📈 Tipo de gráfico:",
            [
                "Gráfico de barras",
                "Gráfico de pizza",
                "Gráfico de pontos",
                "Gráfico de linhas",
                "Ogiva",
                "Boxplot",
            ],
            key=f"grafico_{numero}",
        )

        # ----------------------------------------------------
        # BOTÃO
        # ----------------------------------------------------
        # O estado fica guardado em session_state. Sem isso, qualquer clique
        # (inclusive no botão de baixar o PNG) faz a análise sumir da tela.

        chave_analise = f"analise_ativa_{numero}"

        if st.button("🔎 Gerar análise", key=f"gerar_{numero}"):
            st.session_state[chave_analise] = True

        if not st.session_state.get(chave_analise, False):
            continue

        # ----------------------------------------------------
        # PREPARO DOS DADOS
        # ----------------------------------------------------

        serie_pa = dados[pergunta].to_pandas()
        serie_pa = aplicar_normalizacao(serie_pa, mapa_normalizacao, ativar_normalizacao)

        valores = coverter_numer(serie_pa)
        numero_classes = calcular_numero_classes(valores)

        tabela_frequencia = criar_tabela_resposta(serie_pa)

        if tabela_frequencia.empty:
            st.warning("Esta pergunta não tem respostas para analisar.")
            continue

        # Para os gráficos, as respostas ficam da menos para a mais frequente.
        tabela_grafico = tabela_frequencia.sort_values(by="Frequência absoluta", ascending=True)

        # ----------------------------------------------------
        # 1) GRÁFICO
        # ----------------------------------------------------

        st.markdown("### 📈 Gráfico")

        fig = criar_figura(grafico, tabela_grafico, serie_pa, numero_classes)

        st.plotly_chart(
            fig,
            config={
                "displaylogo": False,
                "scrollZoom": True,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": f"grafico_{numero}",
                    "scale": 2,
                },
            },
            key=f"plotly_{numero}",
            **LARGURA_TOTAL,
        )

        # ----------------------------------------------------
        # 2) DOWNLOAD PNG
        # ----------------------------------------------------

        try:
            imagem_png = figura_png(fig)
            st.download_button(
                label="⬇️ Baixar gráfico PNG",
                data=imagem_png,
                file_name=f"grafico_{numero}.png",
                mime="image/png",
                key=f"download_{numero}",
            )
        except Exception as erro_png:
            st.warning(
                "O gráfico interativo está disponível, mas não foi possível gerar o PNG. "
                "Verifique se a dependência kaleido está instalada. "
                f"Detalhes: {erro_png}"
            )

        # ----------------------------------------------------
        # 3) MEDIDA ESTATÍSTICA + 4) TABELA DE FREQUÊNCIA
        # ----------------------------------------------------

        mostrar_resultado_e_tabela(
            medida,
            quartil,
            valores,
            numero_classes,
            tabela_frequencia,
        )

else:
    st.info("👆 Envie um arquivo Excel para começar.")
