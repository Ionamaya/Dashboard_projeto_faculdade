import streamlit as st
import polars as po
import pandas as pa
import numpy as num
import matplotlib.pyplot as mat
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


def intervalo_classe(valores, numero_de_classe):
    """
    Cria os limites dos intervalos de classe.
    """
    mini = valores.min()
    maxi = valores.max()

    if mini == maxi:
        return None

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
        # TIPO DE CÁLCULO
        # ====================================================

        tipo_calculo = st.selectbox(
            "📊 Tipo de cálculo:",
            [
                "Sem intervalo de classe",
                "Com intervalo de classe"
            ],
            key=f"intervalo_{numero}"
        )


        # ====================================================
        # QUARTIL
        # ====================================================

        quartil = None

        if medida == "Quartil":

            quartil = st.selectbox(
                "Escolha o quartil:",
                [
                    "Q1",
                    "Q2",
                    "Q3"
                ],
                key=f"quartil_{numero}"
            )


        # ====================================================
        # NÚMERO DE CLASSES
        # ====================================================

        numero_classes = 5

        if tipo_calculo == "Com intervalo de classe":

            numero_classes = st.number_input(
                "Número de classes:",
                min_value=2,
                max_value=20,
                value=5,
                step=1,
                key=f"classes_{numero}"
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

            # Converte os dados para valores numericos antes dos calculos
            # Isso garante que a variavel `valores` exista para todas as opcoes.
            valores = coverter_numer(serie_pa)


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

            tabela_frequencia[
                "Frequência relativa (%)"
            ] = (
                tabela_frequencia[
                    "Frequência absoluta"
                ]
                / len(serie_pa)
            ) * 100


            # =================================================
            # RESULTADO ESTATÍSTICO
            # =================================================

            st.markdown(
                "### 📌 Resultado estatístico"
            )


            # =================================================
            # SEM INTERVALO
            # =================================================

            if tipo_calculo == "Sem intervalo de classe":


                # ---------------------------------------------
                # MODA
                # ---------------------------------------------

                if medida == "Moda":

                    moda = (
                        serie_pa
                        .mode()
                    )

                    if len(moda) > 0:

                        st.write(
                            "**Moda:**",
                            ", ".join(
                                map(
                                    str,
                                    moda.tolist()
                                )
                            )
                        )

                    else:

                        st.write(
                            "Não foi possível calcular a moda."
                        )


                # ---------------------------------------------
                # MÉDIA
                # ---------------------------------------------

                elif medida == "Média":

                    if len(valores) == 0:

                        st.warning(
                            "A média só pode ser calculada "
                            "com dados numéricos."
                        )

                    else:

                        media = valores.mean()

                        st.write(
                            f"**Média:** {media:.2f}"
                        )


                # ---------------------------------------------
                # MEDIANA
                # ---------------------------------------------

                elif medida == "Mediana":

                    if len(valores) == 0:

                        st.warning(
                            "A mediana só pode ser calculada "
                            "com dados numéricos."
                        )

                    else:

                        mediana = valores.median()

                        st.write(
                            f"**Mediana:** {mediana:.2f}"
                        )


                # ---------------------------------------------
                # QUARTIL
                # ---------------------------------------------

                elif medida == "Quartil":

                    if len(valores) == 0:

                        st.warning(
                            "Quartis só podem ser calculados "
                            "com dados numéricos."
                        )

                    else:

                        if quartil == "Q1":

                            resultado = valores.quantile(
                                0.25
                            )

                        elif quartil == "Q2":

                            resultado = valores.quantile(
                                0.50
                            )

                        else:

                            resultado = valores.quantile(
                                0.75
                            )

                        st.write(
                            f"**{quartil}:** "
                            f"{resultado:.2f}"
                        )


            # =================================================
            # COM INTERVALO
            # =================================================

            else:

                valores = coverter_numer(
                    serie
                )


                if len(valores) == 0:

                    st.warning(
                        "O intervalo de classe "
                        "só pode ser utilizado "
                        "com dados numéricos."
                    )

                else:

                    tabela_intervalo = (
                        criar_tabela_intervalo(
                            valores,
                            numero_classes
                        )
                    )


                    if tabela_intervalo is not None:

                        (
                            tabela_classes,
                            classe,
                            intervalos
                        ) = tabela_intervalo


                        # -------------------------------------
                        # MODA
                        # -------------------------------------

                        if medida == "Moda":

                            resultado = (
                                calcular_moda_intervalada(
                                    valores,
                                    numero_classes
                                )
                            )

                            if resultado is not None:

                                st.write(
                                    f"**Moda aproximada:** "
                                    f"{resultado:.2f}"
                                )


                        # -------------------------------------
                        # MÉDIA
                        # -------------------------------------

                        elif medida == "Média":

                            resultado = (
                                calcular_media_intervalo(
                                    valores,
                                    numero_classes
                                )
                            )

                            if resultado is not None:

                                st.write(
                                    f"**Média aproximada:** "
                                    f"{resultado:.2f}"
                                )


                        # -------------------------------------
                        # MEDIANA
                        # -------------------------------------

                        elif medida == "Mediana":

                            resultado = (
                                calcular_mediana_intervalo(
                                    valores,
                                    numero_classes
                                )
                            )

                            if resultado is not None:

                                st.write(
                                    f"**Mediana aproximada:** "
                                    f"{resultado:.2f}"
                                )


                        # -------------------------------------
                        # QUARTIL
                        # -------------------------------------

                        elif medida == "Quartil":

                            resultado = (
                                quartil_intervalo(
                                    valores,
                                    numero_classes,
                                    quartil
                                )
                            )

                            if resultado is not None:

                                st.write(
                                    f"**{quartil} aproximado:** "
                                    f"{resultado:.2f}"
                                )


                        # -------------------------------------
                        # TABELA DE CLASSES
                        # -------------------------------------

                        st.markdown(
                            "### 📋 Tabela de frequência "
                            "com intervalo"
                        )

                        st.dataframe(
                            tabela_classes,
                            use_container_width=True
                        )


            # =================================================
            # GRÁFICOS
            # =================================================

            st.markdown(
                "### 📈 Gráfico"
            )

            fig, ax = mat.subplots()


            # =================================================
            # GRÁFICO DE BARRAS
            # =================================================

            if grafico == "Gráfico de barras":

                ax.bar(
                    tabela_frequencia[
                        "Resposta"
                    ],
                    tabela_frequencia[
                        "Frequência absoluta"
                    ]
                )

                ax.set_xlabel(
                    "Resposta"
                )

                ax.set_ylabel(
                    "Frequência"
                )

                ax.set_title(
                    pergunta
                )

                ax.tick_params(
                    axis="x",
                    rotation=45
                )


            # =================================================
            # GRÁFICO DE PIZZA
            # =================================================

            elif grafico == "Gráfico de pizza":

                ax.pie(
                    tabela_frequencia[
                        "Frequência absoluta"
                    ],
                    labels=tabela_frequencia[
                        "Resposta"
                    ],
                    autopct="%1.1f%%"
                )

                ax.set_title(
                    pergunta
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

                ax.set_title(
                    pergunta
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

                ax.set_title(
                    pergunta
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
                    serie
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

                    ax.set_title(
                        f"Ogiva - {pergunta}"
                    )


            # =================================================
            # BOXPLOT
            # =================================================

            elif grafico == "Boxplot":

                valores = coverter_numer(
                    serie
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

                ax.set_title(
                    pergunta
                )


            # =================================================
            # MOSTRAR GRÁFICO
            # =================================================

            st.pyplot(
                fig
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
            # TABELA DE FREQUÊNCIA
            # =================================================

            st.markdown(
                "### 📋 Tabela de frequência"
            )

            st.dataframe(
                tabela_frequencia,
                use_container_width=True
            )

else:

    st.info(
        "👆 Envie um arquivo Excel para começar."
    )