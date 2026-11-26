import base64
import io
import os
import re
import unicodedata
from xml.sax.saxutils import escape

import pdfplumber
import numpy as num
import pandas as pa
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import polars as po
import streamlit as st

# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

# Coloque o arquivo da logo na MESMA PASTA deste dashboard.py,
# com um destes nomes: logo.png, logo.jpg, logo.jpeg ou logo.webp
NOMES_LOGO = ["logo.png", "logo.jpg", "logo.jpeg", "logo.webp"]

# Logos específicas de cada tema (fundo transparente). Se não existirem,
# o dashboard usa o logo.png comum.
LOGO_MODO_CLARO = "logo_modo_claro.png"
LOGO_MODO_ESCURO = "logo_modo_escuro.png"

# Aparece embaixo do título, para você conferir qual arquivo está rodando.
VERSAO_DO_DASHBOARD = "v8"


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

def tema_atual():
    """
    Tema ativo no navegador: "light", "dark" ou None.
    Precisa do Streamlit 1.46 ou mais novo; em versões antigas devolve None.
    """
    try:
        tipo = st.context.theme.type
    except Exception:
        return None
    return tipo if tipo in ("light", "dark") else None


def encontrar_logo(tema=None):
    """
    Procura a logo ao lado deste script. Retorna o caminho ou None.
    Com tema "light" ou "dark" tenta primeiro a logo desse tema;
    se ela não existir, usa o logo.png comum.
    """
    pasta = os.path.dirname(os.path.abspath(__file__))

    candidatos = list(NOMES_LOGO)
    if tema == "dark":
        candidatos = [LOGO_MODO_ESCURO] + candidatos
    elif tema == "light":
        candidatos = [LOGO_MODO_CLARO] + candidatos

    for nome in candidatos:
        caminho = os.path.join(pasta, nome)
        if os.path.exists(caminho):
            return caminho
    return None


def logo_em_html(caminho, largura=150):
    """Logo em HTML, encostada na direita da coluna."""
    extensao = caminho.rsplit(".", 1)[-1].lower()
    tipo = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(extensao, "png")
    with open(caminho, "rb") as arquivo:
        codigo = base64.b64encode(arquivo.read()).decode("ascii")
    return (
        '<div style="text-align:right;">'
        f'<img src="data:image/{tipo};base64,{codigo}" width="{largura}" style="max-width:100%;height:auto;">'
        "</div>"
    )


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


def abrir_analise(chave):
    """Marca a análise da pergunta como aberta."""
    st.session_state[chave] = True


def fechar_analise(chave):
    """Marca a análise da pergunta como fechada (gráfico, medida e tabelas somem)."""
    st.session_state[chave] = False


# ============================================================
# BLOCO 1 — LEITURA DE ARQUIVOS (EXCEL, CSV E PDF)
# ============================================================

def _converter_colunas_numericas(tabela):
    """
    Para CSV e PDF (tudo chega como texto): colunas em que TODAS as respostas
    são números viram colunas numéricas. Aceita vírgula decimal (2,5).
    Se todos os números forem inteiros, a coluna fica como inteira (3 e não 3.0).
    """
    for coluna in tabela.columns:
        original = tabela[coluna]
        preenchidas = int(original.notna().sum())
        if preenchidas == 0:
            continue
        convertida = pa.to_numeric(
            original.astype("object").map(lambda v: v.replace(",", ".") if isinstance(v, str) else v),
            errors="coerce",
        )
        if int(convertida.notna().sum()) == preenchidas:
            if (convertida.dropna() % 1 == 0).all():
                convertida = convertida.astype("Int64")
            tabela[coluna] = convertida
    return tabela


def _pandas_para_polars(tabela):
    """Converte um DataFrame do pandas para polars (via listas, sem depender de versões)."""
    colunas = {}
    for coluna in tabela.columns:
        valores = []
        for v in tabela[coluna].astype("object").tolist():
            if v is None or pa.isna(v):
                valores.append(None)
            elif hasattr(v, "item"):
                valores.append(v.item())
            else:
                valores.append(v)
        colunas[str(coluna)] = valores
    return po.DataFrame(colunas)


def ler_csv(arquivo):
    """Lê CSV com separador (, ou ;) e codificação (UTF-8 ou Latin-1) detectados sozinhos."""
    bruto = arquivo.getvalue()
    try:
        texto = bruto.decode("utf-8-sig")
    except UnicodeDecodeError:
        texto = bruto.decode("latin-1")

    tabela = pa.read_csv(io.StringIO(texto), sep=None, engine="python", dtype=str)
    tabela.columns = [str(c).strip() for c in tabela.columns]
    tabela = _converter_colunas_numericas(tabela)
    return _pandas_para_polars(tabela)


def ler_pdf(arquivo):
    """
    Lê TABELAS dentro de um PDF (a primeira linha da tabela vira o cabeçalho).
    Funciona com PDFs de texto (gerados por Excel, Word, Google Planilhas etc.).
    PDFs escaneados (foto/imagem) não funcionam.
    """
    try:
        import pdfplumber
    except ImportError:
        raise ValueError("Para ler PDF é preciso instalar a biblioteca pdfplumber (veja o requirements.txt).")

    linhas = []
    with pdfplumber.open(io.BytesIO(arquivo.getvalue())) as pdf:
        for pagina in pdf.pages:
            for tabela_pdf in pagina.extract_tables():
                for linha in tabela_pdf:
                    celulas = [
                        (re.sub(r"\s+", " ", str(c)).strip() or None) if c is not None else None
                        for c in linha
                    ]
                    if any(celulas):
                        linhas.append(celulas)

    if len(linhas) < 2:
        raise ValueError(
            "não encontrei nenhuma tabela dentro do PDF. "
            "Use um PDF com a tabela de respostas em texto (não escaneado)."
        )

    cabecalho_bruto = linhas[0]
    nomes = []
    for i, nome in enumerate(cabecalho_bruto, start=1):
        nome = nome or f"Coluna {i}"
        base, contador = nome, 2
        while nome in nomes:
            nome = f"{base} ({contador})"
            contador += 1
        nomes.append(nome)

    total = len(nomes)
    corpo = []
    for linha in linhas[1:]:
        if linha == cabecalho_bruto:  # cabeçalho repetido em cada página
            continue
        corpo.append((linha + [None] * total)[:total])

    tabela = pa.DataFrame(corpo, columns=nomes)
    tabela = _converter_colunas_numericas(tabela)
    return _pandas_para_polars(tabela)


def ler_arquivo(arquivo):
    """Escolhe o leitor certo pela extensão do arquivo enviado."""
    nome = arquivo.name.lower()
    arquivo.seek(0)

    if nome.endswith((".xlsx", ".xls")):
        return po.read_excel(arquivo)
    if nome.endswith(".csv"):
        return ler_csv(arquivo)
    if nome.endswith(".pdf"):
        return ler_pdf(arquivo)
    raise ValueError("formato de arquivo não suportado (use Excel, CSV ou PDF).")


# ============================================================
# BLOCO 2 — DOWNLOADS (EXCEL, CSV E PDF)
# ============================================================

def nome_arquivo_seguro(texto):
    """Transforma um título em nome de arquivo: 'Frequência por resposta' -> 'frequencia_por_resposta'."""
    texto = normalizar_texto(str(texto))
    return re.sub(r"[^a-z0-9]+", "_", texto).strip("_") or "arquivo"


def dataframe_para_csv(tabela):
    """CSV que abre certinho no Excel brasileiro (separador ; e vírgula decimal)."""
    return tabela.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")


def dataframe_para_excel(tabela, nome_aba="Dados"):
    """Arquivo .xlsx em memória."""
    nome_aba = re.sub(r"[\\/*?:\[\]]", "", str(nome_aba))[:31] or "Dados"
    saida = io.BytesIO()
    with pa.ExcelWriter(saida, engine="openpyxl") as escritor:
        tabela.to_excel(escritor, index=False, sheet_name=nome_aba)
    return saida.getvalue()


def _texto_celula_pdf(valor):
    """Números com vírgula decimal; inteiros sem casas."""
    if valor is None:
        return ""
    try:
        if pa.isna(valor):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(valor, (bool, num.bool_)):
        return str(valor)
    if isinstance(valor, (int, num.integer)):
        return str(int(valor))
    if isinstance(valor, (float, num.floating)):
        return f"{float(valor):.2f}".replace(".", ",")
    return str(valor)


def tabela_para_pdf(tabela, titulo):
    """PDF (A4 deitado) com a tabela. Retorna os bytes do arquivo."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    saida = io.BytesIO()
    documento = SimpleDocTemplate(
        saida,
        pagesize=landscape(A4),
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )
    estilos = getSampleStyleSheet()
    estilo_celula = ParagraphStyle("celula", parent=estilos["BodyText"], fontSize=9, leading=11)
    estilo_cabecalho = ParagraphStyle(
        "cabecalho", parent=estilo_celula, textColor=colors.white, fontName="Helvetica-Bold"
    )

    linhas = [[Paragraph(escape(str(c)), estilo_cabecalho) for c in tabela.columns]]
    for registro in tabela.itertuples(index=False):
        linhas.append([Paragraph(escape(_texto_celula_pdf(v)), estilo_celula) for v in registro])

    largura_util = landscape(A4)[0] - 3 * cm
    colunas = len(tabela.columns)
    if colunas > 1:
        primeira = largura_util * 0.34
        demais = (largura_util - primeira) / (colunas - 1)
        larguras = [primeira] + [demais] * (colunas - 1)
    else:
        larguras = [largura_util]

    tabela_pdf = Table(linhas, colWidths=larguras, repeatRows=1)
    tabela_pdf.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0b0b0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    documento.build([
        Paragraph(escape(str(titulo)), estilos["Title"]),
        Spacer(1, 10),
        tabela_pdf,
    ])
    return saida.getvalue()


@st.cache_data(show_spinner=False)
def figura_para_pdf(figura_json):
    """
    PDF do gráfico (precisa da biblioteca kaleido).
    Retorna (bytes, None) se deu certo ou (None, mensagem_de_erro) se não deu.
    O resultado fica em cache para não refazer a cada clique.
    """
    try:
        fig = pio.from_json(figura_json)
        altura = int(fig.layout.height or 500)
        return fig.to_image(format="pdf", width=1000, height=altura), None
    except Exception as erro:
        return None, str(erro)


def botoes_download_tabela(tabela, nome, numero, indice):
    """Três botões embaixo de cada tabela: Excel, CSV e PDF."""
    base = f"pergunta_{numero}_{nome_arquivo_seguro(nome)}"
    chave = f"{numero}_{indice}"
    col_excel, col_csv, col_pdf, _ = st.columns([1, 1, 1, 3])

    with col_excel:
        try:
            st.download_button(
                "⬇️ Excel",
                data=dataframe_para_excel(tabela, nome),
                file_name=f"{base}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_xlsx_{chave}",
            )
        except Exception as erro:
            st.caption(f"Excel indisponível: {erro}")

    with col_csv:
        st.download_button(
            "⬇️ CSV",
            data=dataframe_para_csv(tabela),
            file_name=f"{base}.csv",
            mime="text/csv",
            key=f"dl_csv_{chave}",
        )

    with col_pdf:
        try:
            st.download_button(
                "⬇️ PDF",
                data=tabela_para_pdf(tabela, nome),
                file_name=f"{base}.pdf",
                mime="application/pdf",
                key=f"dl_pdf_{chave}",
            )
        except Exception as erro:
            st.caption(f"PDF indisponível: {erro}")


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


def classificar_moda(tabela_frequencia):
    """
    Descobre a(s) moda(s) e classifica a distribuição.
    Retorna (classificacao, lista_de_modas, frequencia_da_moda).

    - Amodal: nenhuma resposta se repete, ou todas têm a mesma frequência.
    - Unimodal, Bimodal, Trimodal: 1, 2 ou 3 respostas empatadas como mais frequentes.
    - Multimodal: 4 ou mais respostas empatadas como mais frequentes.
    """
    if tabela_frequencia.empty:
        return "Sem respostas", [], 0

    maior = int(tabela_frequencia["Frequência absoluta"].max())
    modas = tabela_frequencia.loc[
        tabela_frequencia["Frequência absoluta"] == maior, "Resposta"
    ].tolist()
    distintas = len(tabela_frequencia)

    if maior == 1 or (distintas > 1 and len(modas) == distintas):
        return "Amodal", [], maior

    nomes = {1: "Unimodal", 2: "Bimodal", 3: "Trimodal"}
    return nomes.get(len(modas), "Multimodal"), modas, maior


def criar_tabela_acumulada(tabela, coluna_rotulo):
    """
    Tabela só com a frequência acumulada (absoluta e em %).
    coluna_rotulo é "Resposta" (tabela por resposta) ou "Intervalo" (por intervalos).
    """
    return tabela[[coluna_rotulo, "Frequência acumulada", "Frequência acumulada (%)"]].copy()


def intervalo_classe(valores, numero_de_classe):
    """Cria os limites dos intervalos de classe."""
    mini = valores.min()
    maxi = valores.max()

    if mini == maxi:
        return num.array([mini - 0.5, maxi + 0.5])

    return num.linspace(mini, maxi, numero_de_classe + 1)


def formatar_numero(x):
    """
    Formata com 1 ou 2 casas depois da vírgula (vírgula decimal).
    Ex.: 1.8 -> 1,8 | 2.0 -> 2,0 | 0.999 -> 1,0 | 10.25 -> 10,25
    """
    x = round(float(x), 2) + 0.0  # evita "-0,0"
    texto = f"{x:.2f}"
    if texto.endswith("0"):
        texto = texto[:-1]
    return texto.replace(".", ",")


def rotulos_intervalos(intervalos):
    """
    Nomes dos intervalos, ex.: [1,0 ; 1,8] e (1,8 ; 2,6].
    O primeiro é fechado nos dois lados, os demais só à direita.
    """
    rotulos = []
    for i in range(len(intervalos) - 1):
        abre = "[" if i == 0 else "("
        rotulos.append(f"{abre}{formatar_numero(intervalos[i])} ; {formatar_numero(intervalos[i + 1])}]")
    return rotulos


def criar_tabela_intervalo(valores, numero_de_classe):
    """Cria uma tabela de frequência com intervalos de classe."""
    intervalos = intervalo_classe(valores, numero_de_classe)

    if intervalos is None:
        return None

    classe = pa.cut(valores, bins=intervalos, include_lowest=True)
    frequencias = classe.value_counts(sort=False)

    tabela = pa.DataFrame({
        "Intervalo": rotulos_intervalos(intervalos),
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

    elif grafico == "Gráfico de colunas":
        tabela_colunas = tabela_grafico.sort_values(by="Frequência absoluta", ascending=False)
        fig = px.bar(
            tabela_colunas,
            x="Resposta",
            y="Frequência absoluta",
            text="Frequência absoluta",
            title="Frequência das respostas",
            labels={"Frequência absoluta": "Frequência", "Resposta": "Resposta"},
            hover_data={"Frequência relativa (%)": ":.2f"},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            xaxis={"categoryorder": "array", "categoryarray": tabela_colunas["Resposta"].tolist()},
            xaxis_tickangle=-35,
            height=500,
            margin=dict(l=20, r=20, t=65, b=90),
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


def criar_histograma(tabela_intervalos, intervalos, acumulado=False):
    """
    Histograma com as barras coladas, uma por intervalo de classe.
    acumulado=False -> frequência absoluta de cada intervalo
    acumulado=True  -> frequência acumulada
    """
    coluna = "Frequência acumulada" if acumulado else "Frequência absoluta"
    titulo = "Histograma — frequência acumulada" if acumulado else "Histograma — frequência por intervalo"
    amplitude = float(intervalos[1] - intervalos[0])

    fig = go.Figure(go.Bar(
        x=tabela_intervalos["Ponto médio"],
        y=tabela_intervalos[coluna],
        width=amplitude,
        text=tabela_intervalos[coluna],
        textposition="outside",
        customdata=tabela_intervalos["Intervalo"],
        hovertemplate="<b>%{customdata}</b><br>Frequência: %{y}<extra></extra>",
        marker=dict(line=dict(width=1, color="white")),
    ))
    fig.update_layout(
        title=titulo,
        xaxis=dict(
            title="Intervalos de classe",
            tickmode="array",
            tickvals=[float(v) for v in intervalos],
            ticktext=[formatar_numero(v) for v in intervalos],
        ),
        yaxis_title="Frequência acumulada" if acumulado else "Frequência",
        bargap=0,
        height=500,
        template="plotly_white",
        font=dict(size=13),
        margin=dict(l=20, r=20, t=65, b=50),
    )
    return fig


def dados_do_grafico(grafico, tabela_grafico, serie_pa, numero_classes):
    """
    Dados que alimentam o gráfico, para baixar em CSV.
    Ogiva -> limites e frequências acumuladas | Boxplot -> os valores numéricos |
    demais gráficos -> a tabela de frequência por resposta.
    """
    if grafico == "Ogiva":
        valores = coverter_numer(serie_pa)
        if len(valores) >= 2:
            tabela_ogiva, _, intervalos = criar_tabela_intervalo(valores, numero_classes)
            acumuladas = tabela_ogiva["Frequência"].cumsum().to_numpy()
            return pa.DataFrame({
                "Limite da classe": num.insert(intervalos[1:], 0, intervalos[0]),
                "Frequência acumulada": num.insert(acumuladas, 0, 0),
            })
    elif grafico == "Boxplot":
        valores = coverter_numer(serie_pa)
        if len(valores) >= 1:
            return pa.DataFrame({"Valores": valores.to_numpy()})

    return tabela_grafico.copy()


# ============================================================
# BLOCO 3 — GRÁFICO NA TELA COM BOTÕES DE DOWNLOAD (CSV E PDF)
# ============================================================

def mostrar_grafico_e_downloads(fig, dados_grafico, nome_base, chave):
    """Mostra o gráfico e, embaixo dele, os botões de baixar CSV e PDF."""
    st.plotly_chart(
        fig,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "toImageButtonOptions": {"format": "png", "filename": nome_base, "scale": 2},
        },
        key=f"plotly_{chave}",
        **LARGURA_TOTAL,
    )

    erro_pdf = None
    col_csv, col_pdf, _ = st.columns([1, 1, 3])

    with col_csv:
        st.download_button(
            "⬇️ Dados do gráfico (CSV)",
            data=dataframe_para_csv(dados_grafico),
            file_name=f"{nome_base}.csv",
            mime="text/csv",
            key=f"dl_grafico_csv_{chave}",
        )

    with col_pdf:
        pdf, erro_pdf = figura_para_pdf(fig.to_json())
        if pdf is not None:
            st.download_button(
                "⬇️ Gráfico (PDF)",
                data=pdf,
                file_name=f"{nome_base}.pdf",
                mime="application/pdf",
                key=f"dl_grafico_pdf_{chave}",
            )

    if erro_pdf:
        st.caption(f"PDF do gráfico indisponível neste servidor ({erro_pdf[:160]}).")


# ------------------------------------------------------------
# RESULTADO ESTATÍSTICO + GRÁFICO + TABELAS DE FREQUÊNCIA
# ------------------------------------------------------------

def mostrar_resultado_e_tabela(
    medida,
    quartil,
    valores,
    numero_classes,
    tabela_frequencia,
    numero,
    grafico,
    tabela_grafico,
    serie_pa,
    area_grafico,
):
    """
    Mostra, nesta ordem na tela:
    1) o gráfico (dentro de area_grafico, que fica reservada lá em cima);
    2) a medida estatística escolhida e o seu resultado;
    3) uma caixa para escolher qual tabela ver;
    4) a(s) tabela(s) escolhida(s), cada uma com botões de download.

    O gráfico é decidido DEPOIS da escolha da tabela:
    - tabela "Frequência por intervalos"            -> Histograma
    - tabela "Frequência acumulada por intervalos"  -> Histograma acumulado
    - qualquer outra escolha                        -> o gráfico escolhido no "Tipo de gráfico"
    """
    tabela_intervalos = None
    intervalos_hist = None

    st.markdown("### 📌 Resultado estatístico")

    if medida == "Moda":
        # A moda usa as respostas originais, então vale para texto e para número.
        classificacao, modas, maior_frequencia = classificar_moda(tabela_frequencia)

        if classificacao == "Amodal":
            motivo = (
                "nenhuma resposta se repete"
                if maior_frequencia == 1
                else "todas as respostas têm a mesma frequência"
            )
            st.write("**Moda:** não teve moda")
            st.write(f"**Classificação:** Amodal ({motivo})")
        else:
            st.write(f"**Moda:** {', '.join(modas)}")
            if len(modas) == 1:
                detalhe = f"1 moda, com {maior_frequencia} respostas"
            else:
                detalhe = f"{len(modas)} modas, com {maior_frequencia} respostas cada"
            st.write(f"**Classificação:** {classificacao} ({detalhe})")

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

    # Se existirem respostas numéricas, a tabela por intervalos também é mostrada
    # (é ela que dá origem à média, mediana e quartis aproximados).
    if len(valores) > 0:
        tabela_intervalos, _, intervalos_hist = criar_tabela_intervalo(valores, numero_classes)
        tabela_intervalos = tabela_intervalos.rename(columns={"Frequência": "Frequência absoluta"})

    # Tabelas disponíveis. As de intervalos só existem quando há respostas numéricas.
    tabelas = [
        (
            "Frequência por resposta",
            tabela_frequencia,
            f"Total de respostas consideradas: {int(tabela_frequencia['Frequência absoluta'].sum())}",
        ),
        (
            "Frequência acumulada por resposta",
            criar_tabela_acumulada(tabela_frequencia, "Resposta"),
            None,
        ),
    ]

    if tabela_intervalos is not None:
        tabelas += [
            (
                "Frequência por intervalos",
                tabela_intervalos,
                f"Total de respostas numéricas consideradas: {int(tabela_intervalos['Frequência absoluta'].sum())}",
            ),
            (
                "Frequência acumulada por intervalos",
                criar_tabela_acumulada(tabela_intervalos, "Intervalo"),
                None,
            ),
        ]

    opcoes = [nome for nome, _, _ in tabelas] + ["Todas as tabelas"]

    escolha = st.selectbox(
        "📋 Qual tabela você deseja ver?",
        opcoes,
        key=f"tabela_{numero}",
    )

    # --------------------------------------------------------
    # GRÁFICO (desenhado na área reservada lá em cima)
    # --------------------------------------------------------
    with area_grafico:
        if escolha in ("Frequência por intervalos", "Frequência acumulada por intervalos"):
            acumulado = escolha == "Frequência acumulada por intervalos"
            coluna_frequencia = "Frequência acumulada" if acumulado else "Frequência absoluta"
            sufixo = "histograma_acumulado" if acumulado else "histograma"

            st.markdown("### 📊 Histograma acumulado" if acumulado else "### 📊 Histograma")
            fig = criar_histograma(tabela_intervalos, intervalos_hist, acumulado=acumulado)
            dados = tabela_intervalos[["Intervalo", "Ponto médio", coluna_frequencia]]
            mostrar_grafico_e_downloads(fig, dados, f"pergunta_{numero}_{sufixo}", f"{numero}_{sufixo}")
        else:
            sufixo = nome_arquivo_seguro(grafico)

            st.markdown("### 📈 Gráfico")
            fig = criar_figura(grafico, tabela_grafico, serie_pa, numero_classes)
            dados = dados_do_grafico(grafico, tabela_grafico, serie_pa, numero_classes)
            mostrar_grafico_e_downloads(fig, dados, f"pergunta_{numero}_{sufixo}", f"{numero}_{sufixo}")

    # --------------------------------------------------------
    # TABELAS (com download em Excel, CSV e PDF)
    # --------------------------------------------------------
    for indice, (nome, tabela, legenda) in enumerate(tabelas):
        if escolha in (nome, "Todas as tabelas"):
            st.markdown(f"### 📋 Tabela de {nome.lower()}")
            tabela_arredondada = tabela.round(2)
            st.dataframe(tabela_arredondada, hide_index=True, **LARGURA_TOTAL)
            if legenda:
                st.caption(legenda)
            botoes_download_tabela(tabela_arredondada, nome, numero, indice)


# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Dashboard - Inotive",
    page_icon="🗃️",
    layout="wide",
)

# ============================================================
# CABEÇALHO COM A LOGO
# ============================================================

caminho_logo = encontrar_logo(tema_atual())

if caminho_logo:
    col_titulo, col_logo = st.columns([4, 1])
    with col_titulo:
        st.title("📊 Dashboard Estatístico - Powered by 創 Inotive")
    with col_logo:
        st.markdown(logo_em_html(caminho_logo), unsafe_allow_html=True)
else:
    st.title("📊 Dashboard Estatístico -Powered by 創 Inotive")
    st.caption("ℹ️ Para exibir a logo, salve o arquivo como logo.png na mesma pasta deste dashboard.py.")

st.caption(f"Versão do dashboard: {VERSAO_DO_DASHBOARD}")

st.write(
    "Envie o arquivo (Excel, CSV ou PDF) com as respostas "
    "do questionário para realizar as análises."
)

# ============================================================
# ENVIO DO ARQUIVO
# ============================================================

arquivo = st.file_uploader(
    "📁 Envie aqui o arquivo escolhido para análise (Excel, CSV ou PDF)",
    type=["xls", "xlsx", "csv", "pdf"],
)

# ============================================================
# LEITURA DO ARQUIVO
# ============================================================

if arquivo is not None:

    try:
        dados = ler_arquivo(arquivo)
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
        if re.match(r"^\s*\d+\s*[.)\-]", pergunta):
            st.subheader(pergunta)  # a pergunta já vem numerada no Excel
        else:
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
                "Gráfico de colunas",
                "Gráfico de pizza",
                "Gráfico de pontos",
                "Gráfico de linhas",
                "Ogiva",
                "Boxplot",
            ],
            key=f"grafico_{numero}",
        )

        # ----------------------------------------------------
        # BOTÕES: GERAR E FECHAR A ANÁLISE
        # ----------------------------------------------------
        # O estado fica guardado em session_state. Sem isso, qualquer clique
        # (inclusive nos botões de download) faz a análise sumir da tela.

        chave_analise = f"analise_ativa_{numero}"

        col_gerar, col_fechar, _ = st.columns([1, 1, 4])

        with col_gerar:
            st.button(
                "🔎 Gerar análise",
                key=f"gerar_{numero}",
                on_click=abrir_analise,
                args=(chave_analise,),
            )

        # O botão de fechar só aparece enquanto a análise está aberta.
        with col_fechar:
            if st.session_state.get(chave_analise, False):
                st.button(
                    "❌ Fechar análise",
                    key=f"fechar_{numero}",
                    on_click=fechar_analise,
                    args=(chave_analise,),
                )

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
        # GRÁFICO + MEDIDA ESTATÍSTICA + TABELAS DE FREQUÊNCIA
        # ----------------------------------------------------
        # A área do gráfico é reservada aqui (em cima) e preenchida lá dentro,
        # depois que a pessoa escolhe a tabela: tabela por intervalos troca o
        # gráfico escolhido pelo histograma.

        area_grafico = st.container()

        mostrar_resultado_e_tabela(
            medida,
            quartil,
            valores,
            numero_classes,
            tabela_frequencia,
            numero,
            grafico,
            tabela_grafico,
            serie_pa,
            area_grafico,
        )

else:
    st.info("👆 Envie um arquivo para começar (Excel, CSV ou PDF).")
