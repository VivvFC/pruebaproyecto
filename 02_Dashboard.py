import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import pearsonr


st.set_page_config(page_title="Quejas Telecomunicaciones", layout="wide")

# CARGA DE DATOS


@st.cache_data
def load_main_data():
    return pd.read_csv(
        "df_telecom_final.csv",
        parse_dates=["fecha_ingreso"]
    )

@st.cache_data
def load_poblacion():
    return pd.read_csv("poblacion_edos.csv")

@st.cache_data
def load_perfiles():
    return pd.read_csv("perfil_proveedores_raw.csv", index_col=0)

@st.cache_data
def load_distancias():
    return pd.read_csv("distancia_euclidiana_proveedores.csv", index_col=0)

@st.cache_data
def load_pca():
    return pd.read_csv("pca_proveedores.csv", index_col=0)

@st.cache_data
def load_mds():
    return pd.read_csv("mds_proveedores.csv", index_col=0)

@st.cache_data
def load_corr():
    return pd.read_csv("correlacion_pearson_proveedores.csv", index_col=0)

df = load_main_data()
df_pob = load_poblacion()

perfil_df = load_perfiles()
dist_df = load_distancias()
pca_df = load_pca()
mds_df = load_mds()
corr_df = load_corr()

st.title("Análisis de Quejas en Telecomunicaciones (PROFECO 2022–2025)")


# PESTAÑAS


tab1, tab2, tab3 = st.tabs([
    "Análisis descriptivo",
    "Análisis económico e inferencial",
    "Análisis multivariado de proveedores"
])


# TAB 1 — DASHBOARD ORIGINAL (BLOQUES 1, 2 y 3)

#  BLOQUE 1 
with tab1:


    st.header("Bloque 1: Comparación de compañías")

    f1, f2, f3, f4 = st.columns(4)

    with f1:
        anios_b1 = st.multiselect(
            "Año",
            sorted(df["anio"].dropna().unique()),
            default=sorted(df["anio"].dropna().unique())
        )

    with f2:
        estados_b1 = st.multiselect(
            "Estado",
            sorted(df["estado"].dropna().unique()),
            default=sorted(df["estado"].dropna().unique())
        )

    with f3:
        proveedores_b1 = st.multiselect(
            "Proveedor",
            sorted(df["proveedor_top"].dropna().unique()),
            default=sorted(df["proveedor_top"].dropna().unique())
        )

    with f4:
        top_n = st.slider("Top N proveedores", 1, 10, 7)

    df_b1 = df.copy()

    if anios_b1:
        df_b1 = df_b1[df_b1["anio"].isin(anios_b1)]
    if estados_b1:
        df_b1 = df_b1[df_b1["estado"].isin(estados_b1)]
    if proveedores_b1:
        df_b1 = df_b1[df_b1["proveedor_top"].isin(proveedores_b1)]

    k1, k2, k3 = st.columns(3)
    k1.metric("Total de quejas", f"{len(df_b1):,}")
    k2.metric("Tasa de conciliación", f"{df_b1['resuelta'].mean()*100:.1f}%")
    k3.metric("Mediana días de resolución", f"{df_b1['dias_resolucion'].median():.0f}")

    top_prov = df_b1["proveedor_top"].value_counts().head(top_n).index

    df_line = (
        df_b1[df_b1["proveedor_top"].isin(top_prov)]
        .groupby([df_b1["fecha_ingreso"].dt.to_period("M"), "proveedor_top"])
        .size()
        .reset_index(name="quejas")
    )
    df_line["fecha_ingreso"] = df_line["fecha_ingreso"].astype(str)

    st.plotly_chart(
        px.line(df_line, x="fecha_ingreso", y="quejas", color="proveedor_top",
                title="Evolución mensual de quejas"),
        use_container_width=True
    )

    df_bar = (
        df_b1.groupby("proveedor_top")
        .agg(total_quejas=("resuelta", "count"),
             conciliadas=("resuelta", "sum"))
        .reset_index()
    )

    df_bar = df_bar.melt(
        id_vars="proveedor_top",
        value_vars=["total_quejas", "conciliadas"],
        var_name="tipo",
        value_name="cantidad"
    )

    st.plotly_chart(
        px.bar(df_bar, x="proveedor_top", y="cantidad",
               color="tipo", barmode="group",
               title="Quejas totales vs conciliadas"),
        use_container_width=True
    )

    c1, c2 = st.columns(2)

    with c1:
        st.plotly_chart(
            px.pie(df_b1, names="tipo_servicio",
                   title="Inconformidades por tipo de servicio"),
            use_container_width=True
        )

    with c2:
        st.plotly_chart(
            px.pie(df_b1, names="estado_procesal",
                   title="Estatus de inconformidades"),
            use_container_width=True
        )

    df_time = (
        df_b1.groupby("proveedor_top")["dias_resolucion"]
        .median()
        .reset_index()
    )

    st.plotly_chart(
        px.bar(df_time, x="proveedor_top", y="dias_resolucion",
               title="Mediana de días de resolución por compañía"),
        use_container_width=True
    )

    #  BLOQUE 2 
    st.header("Bloque 2: Comparación entre estados")

    anio_sel = st.selectbox("Selecciona año", sorted(df["anio"].unique()))
    prov_b2 = st.multiselect(
        "Proveedor",
        sorted(df["proveedor_top"].unique()),
        default=sorted(df["proveedor_top"].unique()),
         key="prov_b2"
    )

    df_b2 = df[(df["anio"] == anio_sel) & (df["proveedor_top"].isin(prov_b2))]

    tabla_estados = df_b2.groupby("estado").size().reset_index(name="quejas")
    tabla_estados = tabla_estados.merge(
        df_pob[["estado", str(anio_sel)]], on="estado", how="left"
    )
    tabla_estados["quejas_100k"] = (
        tabla_estados["quejas"] / tabla_estados[str(anio_sel)] * 100000
    )

    st.dataframe(tabla_estados.sort_values("quejas_100k", ascending=False))

    #  BLOQUE 3 
    st.header("Bloque 3: Motivos de reclamación")

    prov_b3 = st.multiselect(
        "Proveedor",
        sorted(df["proveedor_top"].unique()),
        default=sorted(df["proveedor_top"].unique()),
        key="prov_b3"
    )

    problema_sel = st.selectbox(
        "Problema clasificado",
        sorted(df["problema_clasificado"].unique())
    )

    top_n_motivos = st.slider("Número de motivos a mostrar", 5, 20, 10)

    df_b3 = df[
        (df["proveedor_top"].isin(prov_b3)) &
        (df["problema_clasificado"] == problema_sel)
    ]

    tabla_motivos = (
        df_b3.groupby("motivo_reclamacion")
        .agg(
            frecuencia=("resuelta", "count"),
            tasa_conciliacion=("resuelta", "mean")
        )
        .reset_index()
        .sort_values("frecuencia", ascending=False)
        .head(top_n_motivos)
    )

    tabla_motivos["tasa_conciliacion"] *= 100
    st.dataframe(tabla_motivos)

# TAB 2 — BLOQUE 4: ANÁLISIS ECONÓMICO E INFERENCIAL

with tab2:

    st.header("Análisis económico e inferencial")

    st.markdown(
        """
        Se evalúa la **existencia de relaciones lineales**
        entre variables económicas y operativas mediante:
        
        - Diagramas de dispersión
        - Correlación de Pearson
        - Prueba de hipótesis con t de Student (α = 0.05)
        """
    )

    # Preparamos dataset económico
    eco_df = df[[
        "monto_reclamado",
        "monto_recuperado",
        "resuelta",
        "dias_resolucion"
    ]].dropna()

    eco_df["porcentaje_resolucion"] = eco_df["resuelta"] * 100

    def analizar_correlacion(x, y, x_label, y_label):
        r, p = pearsonr(x, y)

        fig = px.scatter(
            x=x,
            y=y,
            labels={"x": x_label, "y": y_label},
            title=f"{y_label} vs {x_label}"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        **Coeficiente de correlación (r):** {r:.3f}  
        **p-value:** {p:.4f}
        """)

        if p <= 0.05:
            st.success(
                "Con un nivel de significancia de 0.05, se **rechaza la hipótesis nula**. "
                "Existe evidencia estadística de **correlación lineal**."
            )
        else:
            st.warning(
                "Con un nivel de significancia de 0.05, **no se rechaza la hipótesis nula**. "
                "No se encontró evidencia suficiente de correlación lineal."
            )

        st.divider()

    # 1 Monto reclamado vs monto recuperado
    st.subheader("Monto reclamado vs monto recuperado")

    analizar_correlacion(
        eco_df["monto_reclamado"],
        eco_df["monto_recuperado"],
        "Monto reclamado",
        "Monto recuperado"
    )
    # 2 Porcentaje de resolución vs días de resolución
    st.subheader("Monto reclamado vs días de resolución")

    analizar_correlacion(
        eco_df["dias_resolucion"],
        eco_df["monto_reclamado"],
        "Días de resolución",
        "Monto reclamado"
    )

    # 3 Porcentaje de resolución vs días de resolución
    st.subheader("Monto recuperado vs días de resolución")

    analizar_correlacion(
        eco_df["dias_resolucion"],
        eco_df["monto_recuperado"],
        "Días de resolución",
        "Monto reclamado"
    )
    


# TAB 3 — BLOQUE 5: ANÁLISIS MULTIVARIADO


with tab3:

    st.header("Análisis multivariado de proveedores")

    st.markdown(
        """
        En esta sección se utilizan representaciones multivariadas para comparar
        el comportamiento agregado de los principales proveedores de telecomunicaciones.
        Las técnicas empleadas tienen un propósito exploratorio y descriptivo.
        """
    )

    # 1. PERFIL DE PROVEEDORES
    st.subheader("Perfil agregado de proveedores")
    st.markdown(
        "Cada proveedor se representa como un vector numérico que resume su comportamiento promedio."
    )
    st.dataframe(perfil_df)

    # 2. MATRIZ DE DISTANCIAS
    st.subheader("Matriz de distancias euclidianas")
    st.markdown(
        "Valores pequeños indican proveedores con perfiles similares; valores grandes indican mayor disimilitud."
    )

    st.plotly_chart(
        px.imshow(
            dist_df,
            text_auto=".2f",
            title="Distancia euclidiana entre proveedores"
        ),
        use_container_width=True
    )

    # 3. PCA / MDS
    st.subheader("Proyección en dos dimensiones")

    metodo = st.radio(
        "Selecciona método de proyección",
        ["PCA", "MDS"]
    )

    if metodo == "PCA":
        st.markdown(
            "PCA preserva la mayor varianza posible del conjunto de variables originales."
        )
        fig_proj = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            text=pca_df.index,
            title="Proyección PCA de proveedores"
        )
    else:
        st.markdown(
            "MDS preserva las distancias originales entre proveedores."
        )
        fig_proj = px.scatter(
            mds_df,
            x="Dim1",
            y="Dim2",
            text=mds_df.index,
            title="Proyección MDS basada en distancias"
        )

    st.plotly_chart(fig_proj, use_container_width=True)

    # 4. CORRELACIÓN
    st.subheader("Matriz de correlación (Pearson)")
    st.markdown(
        "La correlación de Pearson evalúa relaciones lineales entre variables continuas."
    )

    st.plotly_chart(
        px.imshow(
            corr_df,
            text_auto=".2f",
            title="Correlación de Pearson entre variables"
        ),
        use_container_width=True
    )





