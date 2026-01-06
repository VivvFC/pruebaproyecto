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

# ... (El código anterior de carga de datos y sidebar se mantiene igual)

# =============================================================================
# PESTAÑA 1: ANÁLISIS DESCRIPTIVO
# =============================================================================
    
with tab1:
    st.markdown("### 🔍 Panorama General de Quejas")
        
    # --- FILTROS ESPECÍFICOS DE LA PESTAÑA ---
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        date_range = st.date_input(
            "Selecciona rango de fechas",
            [df_main["fecha_ingreso"].min(), df_main["fecha_ingreso"].max()]
        )
    with col_f2:
        # Filtro opcional de estados para no saturar si no se desea
        all_states = sorted(df_main["estado"].unique())
        selected_states = st.multiselect("Filtrar por Estados (Opcional)", all_states, default=all_states)

    # Aplicar filtros
    mask = (
        (df_main["fecha_ingreso"].dt.date >= date_range[0]) &
        (df_main["fecha_ingreso"].dt.date <= date_range[1])
    )
    if selected_states:
        mask = mask & (df_main["estado"].isin(selected_states))
        
    df_filtered = df_main[mask].copy()

    # --- KPIs ---
    total_quejas = len(df_filtered)
    proveedores_unicos = df_filtered["proveedor"].nunique()
    # Calculamos % de conciliación (asumiendo que existe un estatus 'Conciliada' o similar, ajusta el string según tus datos)
    # Si no tienes una columna exacta para esto, puedes quitar esta métrica o ajustarla
    conciliadas = df_filtered[df_filtered["estatus"].str.contains("Conciliada", case=False, na=False)].shape[0]
    pct_conciliacion = (conciliadas / total_quejas * 100) if total_quejas > 0 else 0

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total de Quejas", f"{total_quejas:,}")
    kpi2.metric("Proveedores Involucrados", f"{proveedores_unicos}")
    kpi3.metric("% Conciliación Aprox.", f"{pct_conciliacion:.1f}%")
    st.markdown("---")

    # --- BLOQUE 1: EVOLUCIÓN (La que te gusta) ---
    # Agrupamos por mes para ver la tendencia
    df_evo = df_filtered.set_index("fecha_ingreso").resample("M").size().reset_index(name="conteo")
        
    fig_evo = px.line(
        df_evo, 
        x="fecha_ingreso", 
        y="conteo", 
        markers=True,
        title="📈 Evolución Mensual de Quejas",
        labels={"fecha_ingreso": "Fecha", "conteo": "Número de Quejas"}
     )
    fig_evo.update_layout(xaxis_title=None)
    st.plotly_chart(fig_evo, use_container_width=True)

    # --- BLOQUE 2: MAPA NORMALIZADO Y TREEMAP ---
    row2_col1, row2_col2 = st.columns([1, 1])

    with row2_col1:
        st.subheader("Geografía del Reclamo")
        # Preparación de datos para el mapa: Cruce con Población
        quejas_edo = df_filtered["estado"].value_counts().reset_index()
        quejas_edo.columns = ["estado", "quejas"]
            
        # Asegúrate que los nombres de estados coincidan entre df_main y df_poblacion
        # Hacemos un merge left
        df_mapa = pd.merge(quejas_edo, df_pob, left_on="estado", right_on="estado", how="left")
            
        # Cálculo de Tasa por 100k habitantes (Más valioso que el conteo simple)
        # Si hay algún estado que no cruzó (NaN), rellenamos con 0 para evitar error
        df_mapa["poblacion"] = df_mapa["poblacion"].fillna(1) 
        df_mapa["tasa_100k"] = (df_mapa["quejas"] / df_mapa["poblacion"]) * 100000

        fig_map = px.choropleth(
            df_mapa,
            geojson="https://raw.githubusercontent.com/angelnmara/geojson/master/mexico_high.json", # GeoJSON público de México
            locations="estado",
            featureidkey="properties.name",
            color="tasa_100k",
            color_continuous_scale="Reds",
            title="Quejas por cada 100k Habitantes (Intensidad Real)",
            hover_data=["quejas", "poblacion"]
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig_map, use_container_width=True)
        st.caption("*El mapa muestra la densidad de quejas relativa a la población, revelando dónde el problema es más agudo proporcionalmente.*")

    with row2_col2:
        st.subheader("Composición de Proveedores")
        # Filtramos Top 15 para que el gráfico sea legible
        top_prov_list = df_filtered["proveedor"].value_counts().nlargest(15).index
        df_treemap = df_filtered[df_filtered["proveedor"].isin(top_prov_list)]
            
        # Treemap: Proveedor -> Estatus (o Medio de Ingreso si prefieres)
        # Esto muestra quién tiene más quejas Y cómo las están manejando
        fig_tree = px.treemap(
            df_treemap,
            path=[px.Constant("Todas"), "proveedor", "estatus"],
            title="Top 15 Proveedores y Estatus de Queja",
            color="proveedor" 
        )
        fig_tree.update_traces(root_color="lightgrey")
        st.plotly_chart(fig_tree, use_container_width=True)

    # --- BLOQUE 3: ANÁLISIS CRUZADO (HEATMAP) ---
    st.subheader("🔥 Focos Rojos: Proveedores vs. Estados")
    st.markdown("Identifica si un proveedor tiene fallas generalizadas o problemas localizados en ciertos estados.")

    # Top 10 proveedores y Top 10 estados con más quejas para el heatmap
    top_p = df_filtered["proveedor"].value_counts().nlargest(10).index
    top_e = df_filtered["estado"].value_counts().nlargest(10).index
        
    df_heat = df_filtered[
        (df_filtered["proveedor"].isin(top_p)) & 
        (df_filtered["estado"].isin(top_e))
    ]
        
    # Crear matriz para heatmap
    heatmap_data = pd.crosstab(df_heat["proveedor"], df_heat["estado"])
        
    fig_heat = px.imshow(
    heatmap_data,
    labels=dict(x="Estado", y="Proveedor", color="N° Quejas"),
    x=heatmap_data.columns,
    y=heatmap_data.index,
    text_auto=True, # Muestra los números dentro de los cuadros
    aspect="auto",
    color_continuous_scale="Viridis"
)
    st.plotly_chart(fig_heat, use_container_width=True)

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









