import streamlit as st
import polars as po
import pandas as pa
import numpy as num
import matplotlib.pyplot as mat
import re
import unicodedata

from io import BytesIO as by

# ============================================================

# FUNÇÕES DE AUXÍLIO

# ============================================================

def coverter_numer(serie):

          """

          Converte uma série para números.

          Valores que não forem numéricos são ignorados.

          """

          valores = pa.to_numeric(

                    serie,

                    errors="coerce"

          )

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

def intervalo_classe(valores, numero_de_classe):

          """

          Cria os limites dos intervalos de classe.

          """

          mini = valores.min()

          maxi = valores.max()

          if mini == maxi:

                    return num.array([mini - 0.5, maxi + 0.5])

          intervalos = num.linspace(

                    mini,

                    maxi,

                    numero_de_classe + 1

          )

          return intervalos

def criar_tabela_intervalo(valores, numero_de_classe):

          """

          Cria uma tabela de frequência com intervalos de classe.

          """

          intervalos = intervalo_classe(

                    valores,

                    numero_de_classe

          )

          if intervalos is None:

                    return None

          classe = pa.cut(

                    valores,

                    bins=intervalos,

                    include_lowest=True

          )

          frequencias = classe.value_counts(

                    sort=False

          )

          tabela = pa.DataFrame({

                    "Intervalo": frequencias.index.astype(str),

                    "Frequência": frequencias.values

          })

          tabela["Ponto médio"] = (

                    intervalos[:-1] + intervalos[1:]

          ) / 2

          tabela["Frequência relativa (%)"] = (

                    tabela["Frequência"]

                    / tabela["Frequência"].sum()

          ) * 100

          tabela["Frequência acumulada"] = (

                    tabela["Frequência"].cumsum()

          )

          return tabela, classe, intervalos

def calcular_media_intervalo(

          valores,

          numero_de_classe

):

          """

          Calcula a média aproximada utilizando

          os pontos médios dos intervalos.

          """

          resultado = criar_tabela_intervalo(

                    valores,

                    numero_de_classe

          )

          if resultado is None:

                    return None

          tabela, classe, intervalos = resultado

          pontos_medios = (

                    intervalos[:-1] +

                    intervalos[1:]

          ) / 2

          frequencias = tabela["Frequência"].values

          media = (

                    pontos_medios * frequencias

          ).sum() / frequencias.sum()

          return media

def calcular_mediana_intervalo(

          valores,

          numero_de_classe

):

          """

          Calcula a mediana aproximada para

          dados agrupados em intervalos.

          """

          resultado = criar_tabela_intervalo(

                    valores,

                    numero_de_classe

          )

          if resultado is None:

                    return None

          tabela, classe, intervalos = resultado

          frequencias = tabela["Frequência"].values

          total = frequencias.sum()

          posicao = total / 2

          acumuladas = num.cumsum(

                    frequencias

          )

          indice = num.where(

                    acumuladas >= posicao

          )[0][0]

          frequencia_classe = frequencias[indice]

          acumuladas_anterior = (

                    acumuladas[indice - 1]

                    if indice > 0

                    else 0

          )

          limite_inferior = intervalos[indice]

          amplitude = (

                    intervalos[indice + 1] -

                    intervalos[indice]

          )

          mediana = (

                    limite_inferior +

                    (

                              (posicao - acumuladas_anterior)

                              / frequencia_classe

                    ) * amplitude

          )

          return mediana

def quartil_intervalo(

          valores,

          numero_de_classe,

          quartil

):

          """

          Calcula Q1, Q2 ou Q3 para dados

          agrupados em intervalos.

          """

          resultado = criar_tabela_intervalo(

                    valores,

                    numero_de_classe

          )

          if resultado is None:

                    return None

          tabela, classe, intervalos = resultado

          frequencias = tabela["Frequência"].values

          total = frequencias.sum()

          if quartil == "Q1":

                    posicao = total / 4

          elif quartil == "Q2":

                    posicao = total / 2

          else:

                    posicao = (3 * total) / 4

          acumuladas = num.cumsum(

                    frequencias

          )

          indice = num.where(

                    acumuladas >= posicao

          )[0][0]

          frequencia_classe = frequencias[indice]

          acumuladas_anterior = (

                    acumuladas[indice - 1]

                    if indice > 0

                    else 0

          )

          limite_inferior = intervalos[indice]

          amplitude = (

                    intervalos[indice + 1] -

                    intervalos[indice]

          )

          resultado_quartil = (

                    limite_inferior +

                    (

                              (posicao - acumuladas_anterior)

                              / frequencia_classe

                    ) * amplitude

          )

          return resultado_quartil

def calcular_moda_intervalada(

          valores,

          numero_de_classe

):

          """

          Calcula a moda aproximada para

          dados agrupados em intervalos.

          """

          resultado = criar_tabela_intervalo(

                    valores,

                    numero_de_classe

          )

          if resultado is None:

                    return None

          tabela, classe, intervalos = resultado

          frequencias = tabela["Frequência"].values

          indice = num.argmax(

                    frequencias

          )

          frequencia_modal = frequencias[indice]

          frequencia_anterior = (

                    frequencias[indice - 1]

                    if indice > 0

                    else 0

          )

          frequencia_posterior = (

                    frequencias[indice + 1]

                    if indice < len(frequencias) - 1

                    else 0

          )

          limite_inferior = intervalos[indice]

          amplitude = (

                    intervalos[indice + 1] -

                    intervalos[indice]

          )

          diferenca_anterior = (

                    frequencia_modal -

                    frequencia_anterior

          )

          diferenca_posterior = (

                    frequencia_modal -

                    frequencia_posterior

          )

          denominador = (

                    diferenca_anterior +

                    diferenca_posterior

          )

          if denominador == 0:

                    return (

                              intervalos[indice] +

                              amplitude / 2

                    )

          moda = (

                    limite_inferior +

                    (

                              diferenca_anterior /

                              denominador

                    ) * amplitude

          )

          return moda

def figura_png(fig):

          """

          Converte o gráfico para PNG.

          """

          imagem = by()

          fig.savefig(

                    imagem,

                    format="png",

                    bbox_inches="tight"

          )

          imagem.seek(0)

          return imagem

# ============================================================

# CONFIGURAÇÃO DA PÁGINA

# ============================================================

st.set_page_config(

          page_title="Dashboard de Super Mercados",

          page_icon="🗃️",

          layout="wide"

)

# ============================================================

# ENVIO DO ARQUIVO

# ============================================================

st.title("📊 Dashboard Estatístico - Supermercados")

st.write(

          "Envie o arquivo Excel com as respostas "

          "do questionário para realizar as análises."

)

arquivo = st.file_uploader(

          "📁 Envie aqui o arquivo Excel escolhido para análise",

          type=["xls", "xlsx"]

)

# ============================================================

# LEITURA DO ARQUIVO

# ============================================================

if arquivo is not None:

          try:

                    dados = po.read_excel(

                              arquivo

                    )

                    st.success(

                              "✅ Arquivo carregado com sucesso!"

                    )

          except Exception as erro:

                    st.error(

                              f"❌ Não foi possível ler o arquivo: {erro}"

                    )

                    st.stop()

          # ========================================================

          # DADOS

          # ========================================================

          st.subheader("📋 Dados")

          st.dataframe(

                    dados,

                    use_container_width=True

          )

          st.write(

                    "Quantidade de linhas:",

                    dados.height

          )

          st.write(

                    "Quantidade de colunas:",

                    dados.width

          )

          # ========================================================

          # LISTA DE PERGUNTAS

          # ========================================================

          perguntas = [

                    coluna

                    for coluna in dados.columns

                    if coluna.lower()

                    not in [

                              "carimbo de data/hora",

                              "timestamp"

                    ]

          ]

          # ========================================================

          # PADRONIZAÇÃO DAS RESPOSTAS TEXTUAIS

          # ========================================================

          st.subheader("🧹 Padronização das respostas")

          st.write("O filtro reúne diferenças de maiúsculas, minúsculas, acentos e espaços. Você também pode cadastrar abreviações e diminutivos.")

          ativar_normalizacao = st.checkbox("Ativar padronização automática", value=True, key="ativar_normalizacao")

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

                              key="regras_normalizacao_v2"

                    )

          mapa_normalizacao = criar_mapa_normalizacao(texto_mapeamento)

          # ========================================================

          # ANÁLISE DAS PERGUNTAS

          # ========================================================

          st.header(

                    "📊 Análise das perguntas"

          )

          for numero, pergunta in enumerate(

                    perguntas,

                    start=1

          ):

                    st.divider()

                    st.subheader(

                              f"{numero}. {pergunta}"

                    )

                    # ====================================================

                    # MEDIDA ESTATÍSTICA

                    # ====================================================

                    medida = st.selectbox(

                              "📐 Medida estatística:",

                              [

                                        "Moda",

                                        "Média",

                                        "Mediana",

                                        "Quartil"

                              ],

                              key=f"medida_{numero}"

                    )

                    # ====================================================

                    # QUARTIL

                    # ====================================================

                    quartil = None

                    if medida == "Quartil":

                              quartil = st.selectbox(

                                        "Escolha o quartil:",

                                        ["Q1", "Q2", "Q3"],

                                        key=f"quartil_{numero}"

                              )

                    # ====================================================

                    # ESCOLHA DO GRÁFICO

                    # ====================================================

                    grafico = st.selectbox(

                              "📈 Tipo de gráfico:",

                              [

                                        "Gráfico de barras",

                                        "Gráfico de pizza",

                                        "Gráfico de pontos",

                                        "Gráfico de linhas",

                                        "Ogiva",

                                        "Boxplot"

                              ],

                              key=f"grafico_{numero}"

                    )

                    # ====================================================

                    # BOTÃO

                    # ====================================================

                    gerar = st.button(

                              "🔎 Gerar análise",

                              key=f"gerar_{numero}"

                    )

                    # ====================================================

                    # GERAR ANÁLISE

                    # ====================================================

                    if gerar:

                              serie = dados[pergunta]

                              serie_pa = serie.to_pandas()

                              serie_pa = aplicar_normalizacao(

                                        serie_pa, mapa_normalizacao, ativar_normalizacao

                              )

                              # Converte os dados para valores numericos antes dos calculos

                              # Isso garante que a variavel \\\\\\\`valores\\\\\\\` exista para todas as opcoes.

                              valores = coverter_numer(serie_pa)

                              # Número de classes calculado automaticamente (regra de Sturges)

                              if len(valores) >= 2:

                                        numero_classes = max(

                                                  2,

                                                  min(20, int(num.ceil(1 + 3.322 * num.log10(len(valores)))))

                                        )

                              else:

                                        numero_classes = 2

                              # =================================================

                              # =================================================
                              # TABELA DE FREQUÊNCIA

                              # =================================================

                              frequencias = (

                                        serie_pa

                                        .value_counts()

                                        .sort_index()

                              )

                              tabela_frequencia = pa.DataFrame({

                                        "Resposta":

                                                  frequencias.index.astype(str),

                                        "Frequência absoluta":

                                                  frequencias.values

                              })
                              tabela_frequencia = tabela_frequencia.sort_values(
                                        by="Frequência absoluta",
                                        ascending=True
                              )

                              tabela_frequencia[

                                        "Frequência relativa (%)"

                              ] = (

                                        tabela_frequencia[

                                                  "Frequência absoluta"

                                        ]

                                        / serie_pa.notna().sum()

                              ) * 100

                              # =================================================

                              # GRÁFICOS

                              # =================================================

                              st.markdown(

                                        "### 📈 Gráfico"

                              )

                              fig, ax = mat.subplots(figsize=(7, 4), dpi=100)

                              # =================================================

                              # GRÁFICO DE BARRAS

                              # =================================================

                              if grafico == "Gráfico de barras":

                                        # Remove respostas vazias

                                        tabela_grafico = tabela_frequencia[

                                                  tabela_frequencia["Resposta"].astype(str).str.strip() != ""

                                        ].copy()

                                        # Ordena as frequências

                                        tabela_grafico = tabela_grafico.sort_values(

                                                  by="Frequência absoluta",

                                                  ascending=True

                                        )

                                        quantidade_respostas = len(tabela_grafico)

                                        # Ajusta a altura conforme a quantidade de categorias

                                        altura = max(4, quantidade_respostas * 0.45)

                                        fig.set_size_inches(9, altura)

                                        # Cria o gráfico horizontal

                                        ax.barh(

                                                  tabela_grafico["Resposta"].astype(str),

                                                  tabela_grafico["Frequência absoluta"]

                                        )

                                        ax.set_xlabel("Frequência")

                                        ax.set_ylabel("Resposta")

                                        ax.tick_params(

                                                  axis="y",

                                                  labelsize=9

                                        )

                                        # Calcula o espaço necessário para os nomes

                                        maior_nome = max(

                                                  (len(str(nome)) for nome in tabela_grafico["Resposta"]),

                                                  default=10

                                        )

                                        margem_esquerda = min(

                                                  0.65,

                                                  max(0.30, maior_nome * 0.009)

                                        )

                                        fig.subplots_adjust(

                                                  left=margem_esquerda,

                                                  right=0.96,

                                                  top=0.96,

                                                  bottom=0.12

                                        )

                              # =================================================

                              # GRÁFICO DE PIZZA

                              # =================================================

                              elif grafico == "Gráfico de pizza":

                                        # Remove respostas vazias

                                        tabela_grafico = tabela_frequencia[

                                                  tabela_frequencia["Resposta"].astype(str).str.strip() != ""

                                        ].copy()

                                        fig.set_size_inches(9, 5.5)

                                        # Cria o gráfico sem nomes ao redor das fatias

                                        fatias, textos, porcentagens = ax.pie(

                                                  tabela_grafico["Frequência absoluta"],

                                                  labels=None,

                                                  autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",

                                                  startangle=90,

                                                  pctdistance=0.70,

                                                  textprops={

                                                            "fontsize": 9

                                                  }

                                        )

                                        # Coloca os nomes em uma legenda separada

                                        ax.legend(

                                                  fatias,

                                                  tabela_grafico["Resposta"].astype(str),

                                                  title="Respostas",

                                                  loc="center left",

                                                  bbox_to_anchor=(1.02, 0.5),

                                                  fontsize=8,

                                                  title_fontsize=9,

                                                  frameon=False

                                        )

                                        ax.set_aspect("equal")

                                        fig.subplots_adjust(

                                                  left=0.02,

                                                  right=0.65,

                                                  top=0.95,

                                                  bottom=0.08

                                        )

                              # =================================================

                              # GRÁFICO DE PONTOS

                              # =================================================

                              elif grafico == "Gráfico de pontos":

                                        x = range(

                                                  len(tabela_frequencia)

                                        )

                                        ax.scatter(

                                                  x,

                                                  tabela_frequencia[

                                                            "Frequência absoluta"

                                                  ]

                                        )

                                        ax.set_xticks(

                                                  list(x)

                                        )

                                        ax.set_xticklabels(

                                                  tabela_frequencia[

                                                            "Resposta"

                                                  ],

                                                  rotation=45,

                                                  ha="right"

                                        )

                                        ax.set_xlabel(

                                                  "Resposta"

                                        )

                                        ax.set_ylabel(

                                                  "Frequência"

                                        )

                              # =================================================

                              # GRÁFICO DE LINHAS

                              # =================================================

                              elif grafico == "Gráfico de linhas":

                                        ax.plot(

                                                  tabela_frequencia[

                                                            "Resposta"

                                                  ],

                                                  tabela_frequencia[

                                                            "Frequência absoluta"

                                                  ],

                                                  marker="o"

                                        )

                                        ax.set_xlabel(

                                                  "Resposta"

                                        )

                                        ax.set_ylabel(

                                                  "Frequência"

                                        )

                                        ax.tick_params(

                                                  axis="x",

                                                  rotation=45

                                        )

                              # =================================================

                              # OGIVA

                              # =================================================

                              elif grafico == "Ogiva":

                                        valores = coverter_numer(

                                                  serie_pa

                                        )

                                        if len(valores) < 2:

                                                  mat.close(fig)

                                                  st.warning(

                                                            "A ogiva precisa de pelo menos "

                                                            "dois dados numéricos."

                                                  )

                                                  continue

                                        resultado = (

                                                  criar_tabela_intervalo(

                                                            valores,

                                                            numero_classes

                                                  )

                                        )

                                        if resultado is not None:

                                                  (

                                                            tabela_ogiva,

                                                            classe,

                                                            intervalos

                                                  ) = resultado

                                                  frequencias_acumuladas = (

                                                            tabela_ogiva[

                                                                      "Frequência"

                                                            ].cumsum()

                                                  )

                                                  limites_superiores = (

                                                            intervalos[1:]

                                                  )

                                                  x_ogiva = num.insert(

                                                            limites_superiores,

                                                            0,

                                                            intervalos[0]

                                                  )

                                                  y_ogiva = num.insert(

                                                            frequencias_acumuladas.values,

                                                            0,

                                                            0

                                                  )

                                                  ax.plot(

                                                            x_ogiva,

                                                            y_ogiva,

                                                            marker="o"

                                                  )

                                                  ax.set_xlabel(

                                                            "Limite da classe"

                                                  )

                                                  ax.set_ylabel(

                                                            "Frequência acumulada"

                                                  )

                              # =================================================

                              # BOXPLOT

                              # =================================================

                              elif grafico == "Boxplot":

                                        valores = coverter_numer(

                                                  serie_pa

                                        )

                                        if len(valores) < 1:

                                                  mat.close(fig)

                                                  st.warning(

                                                            "O boxplot precisa de "

                                                            "dados numéricos."

                                                  )

                                                  continue

                                        ax.boxplot(

                                                  valores

                                        )

                                        ax.set_ylabel(

                                                  "Valores"

                                        )

                              # =================================================

                              # MOSTRAR GRÁFICO

                              # =================================================

                              st.pyplot(

                                        fig,

                                        width=700

                              )

                              # =================================================

                              # DOWNLOAD PNG

                              # =================================================

                              imagem_png = figura_png(

                                        fig

                              )

                              nome_arquivo = (

                                        f"grafico_{numero}.png"

                              )

                              st.download_button(

                                        label="⬇️ Baixar gráfico PNG",

                                        data=imagem_png,

                                        file_name=nome_arquivo,

                                        mime="image/png",

                                        key=f"download_{numero}"

                              )

                              mat.close(

                                        fig

                              )

                              # =================================================

                              # RESULTADO ESTATÍSTICO
                              # =================================================
                              st.markdown("### 📌 Resultado estatístico")

                              # A MODA é calculada pelas respostas originais,
                              # então funciona tanto para textos quanto para números.
                              if medida == "Moda":
                                        frequencias_moda = serie_pa.dropna().value_counts()

                                        if frequencias_moda.empty or frequencias_moda.max() == 1:
                                                  st.write("**Moda:** Não existe moda (nenhuma resposta se repete).")
                                        else:
                                                  maior_frequencia = frequencias_moda.max()
                                                  modas = frequencias_moda[frequencias_moda == maior_frequencia].index.tolist()
                                                  resultado_moda = ", ".join(str(moda) for moda in modas)
                                                  st.write(f"**Moda:** {resultado_moda}")

                                        tabela_exibicao = tabela_frequencia
                                        titulo_tabela = "### 📋 Tabela de frequência por resposta"

                              # Média, mediana e quartis precisam de dados numéricos.
                              else:
                                        if len(valores) == 0:
                                                  st.warning("Esta medida precisa de respostas numéricas. Escolha Moda para analisar respostas textuais.")
                                                  tabela_exibicao = tabela_frequencia
                                                  titulo_tabela = "### 📋 Tabela de frequência por resposta"
                                        else:
                                                  resultado_intervalo = criar_tabela_intervalo(valores, numero_classes)

                                                  if resultado_intervalo is None:
                                                            st.warning("Não foi possível formar os intervalos para estes dados.")
                                                            tabela_exibicao = tabela_frequencia
                                                            titulo_tabela = "### 📋 Tabela de frequência por resposta"
                                                  else:
                                                            tabela_classes, classe, intervalos = resultado_intervalo

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

                                                            tabela_exibicao = tabela_classes
                                                            titulo_tabela = "### 📋 Tabela de frequência por intervalos"

                              # TABELA DE FREQUÊNCIA

                              # =================================================

                              st.markdown(titulo_tabela)
                              st.dataframe(
                                        tabela_exibicao,
                                        use_container_width=True
                              )

else:

          st.info(

                    "👆 Envie um arquivo Excel para começar."

          )