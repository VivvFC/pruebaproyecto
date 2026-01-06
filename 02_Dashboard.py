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

# =============================================================================
# PESTAÑA 1: ANÁLISIS DESCRIPTIVO (VERSIÓN CORREGIDA FINAL)
# =============================================================================

with tab1:
    st.markdown("### 🔍 Panorama General de Quejas")
    
    # --- SECCIÓN DE FILTROS ---
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        # Filtro 1: Rango de Fechas
        f_min = df_main["fecha_ingreso"].min()
        f_max = df_main["fecha_ingreso"].max()
        date_range = st.date_input("Rango de Fechas", [f_min, f_max])
    
    with col_f2:
        # Filtro 2: Estados (Multiselect)
        all_states = sorted(df_main["estado"].unique())
        selected_states = st.multiselect("Filtrar por Estados", all_states, default=all_states)

    # --- APLICACIÓN DE FILTROS AL DATAFRAME ---
    # 1. Filtro de Fecha
    if len(date_range) == 2:
        mask_date = (df_main["fecha_ingreso"].dt.date >= date_range[0]) & (df_main["fecha_ingreso"].dt.date <= date_range[1])
        df_filtered = df_main[mask_date].copy()
    else:
        df_filtered = df_main.copy()

    # 2. Filtro de Estado
    if selected_states:
        df_filtered = df_filtered[df_filtered["estado"].isin(selected_states)]

    # --- CÁLCULO DE KPIs ---
    total_quejas = len(df_filtered)
    proveedores_unicos = df_filtered["nombre_comercial"].nunique()
    
    # Cálculo de Conciliación buscando la palabra "Conciliada" en 'estado_procesal'
    # Ajusta "Conciliada" si en tu CSV aparece como "CONCILIADA" (mayúsculas) o diferente.
    # El 'case=False' ayuda a ignorar mayúsculas/minúsculas.
    try:
        quejas_conciliadas = df_filtered[
            df_filtered["estado_procesal"].astype(str).str.contains("Conciliada", case=False, na=False)
        ].shape[0]
        
        pct_conciliacion = (quejas_conciliadas / total_quejas * 100) if total_quejas > 0 else 0
    except KeyError:
        st.error("No se encontró la columna 'estado_procesal'. Verifica el nombre en tu CSV.")
        pct_conciliacion = 0

    # Visualización de KPIs
    k1, k2, k3 = st.columns(3)
    k1.metric("Total de Quejas", f"{total_quejas:,}")
    k2.metric("Proveedores en Filtro", f"{proveedores_unicos}")
    k3.metric("% Conciliación", f"{pct_conciliacion:.1f}%")

    st.markdown("---")

    # --- GRÁFICA 1: EVOLUCIÓN MENSUAL (Por Proveedor y Estado Filtrado) ---
    # Esta gráfica muestra la evolución de los proveedores dentro de los estados seleccionados.
    
    # Agrupamos por Mes y por Nombre Comercial
    df_evo = df_filtered.set_index("fecha_ingreso").groupby(
        [pd.Grouper(freq="M"), "nombre_comercial"]
    ).size().reset_index(name="conteo")

    # Filtro de Top 10 para evitar saturación visual si hay muchas empresas
    top_10_prov = df_evo.groupby("nombre_comercial")["conteo"].sum().nlargest(10).index
    df_evo_final = df_evo[df_evo["nombre_comercial"].isin(top_10_prov)]

    fig_evo = px.line(
        df_evo_final, 
        x="fecha_ingreso", 
        y="conteo", 
        color="nombre_comercial", # <--- Líneas separadas por proveedor
        markers=True,
        title="📈 Evolución Mensual de Quejas por Proveedor (Top 10)",
        labels={"fecha_ingreso": "Fecha", "conteo": "Quejas", "nombre_comercial": "Proveedor"}
    )
    st.plotly_chart(fig_evo, use_container_width=True)

    # --- GRÁFICA 2 y 3: MAPA NORMALIZADO y TREEMAP ---
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.subheader("Mapa de Calor (Tasa por 100k hab.)")
        try:
            # 1. Agrupar datos filtrados por estado
            quejas_edo = df_filtered["estado"].value_counts().reset_index()
            quejas_edo.columns = ["estado", "quejas"]
            
            # 2. Unir con dataframe de población (df_pob)
            # IMPORTANTE: Asegúrate que df_pob tenga columnas 'estado' y 'poblacion'
            df_mapa = pd.merge(quejas_edo, df_pob, on="estado", how="left")
            
            # 3. Calcular tasa
            df_mapa["tasa"] = (df_mapa["quejas"] / df_mapa["poblacion"]) * 100000
            
            fig_map = px.choropleth(
                df_mapa,
                geojson="https://raw.githubusercontent.com/angelnmara/geojson/master/mexico_high.json",
                locations="estado",
                featureidkey="properties.name",
                color="tasa",
                color_continuous_scale="Reds",
                title="Quejas por cada 100k habitantes",
                hover_data=["quejas", "poblacion"]
            )
            fig_map.update_geos(fitbounds="locations", visible=False)
            fig_map.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
            st.plotly_chart(fig_map, use_container_width=True)
        
        except Exception as e:
            st.warning("No se pudo generar el mapa. Revisa que df_pob esté cargado y los nombres de estado coincidan.")
            # Fallback: Mapa simple sin normalizar si falla la población
            fig_map_simple = px.choropleth(
                quejas_edo,
                geojson="https://raw.githubusercontent.com/angelnmara/geojson/master/mexico_high.json",
                locations="estado",
                featureidkey="properties.name",
                color="quejas",
                title="Total de Quejas (Sin normalizar)"
            )
            fig_map_simple.update_geos(fitbounds="locations", visible=False)
            st.plotly_chart(fig_map_simple, use_container_width=True)

    with row2_col2:
        st.subheader("Composición: Proveedor y Estatus")
        # Usamos 'nombre_comercial' y 'estado_procesal'
        top_prov_tree = df_filtered["nombre_comercial"].value_counts().head(10).index
        df_tree = df_filtered[df_filtered["nombre_comercial"].isin(top_prov_tree)]
        
        fig_tree = px.treemap(
            df_tree,
            path=[px.Constant("Total"), "nombre_comercial", "estado_procesal"],
            title="Top 10 Proveedores y su Estatus"
        )
        st.plotly_chart(fig_tree, use_container_width=True)

    # --- GRÁFICA 4: HEATMAP (Matriz de Riesgo) ---
    st.subheader("🔥 Concentración: Proveedor vs Estado")
    
    # Filtramos Top 10 para que el heatmap sea legible
    top_p_heat = df_filtered["nombre_comercial"].value_counts().head(10).index
    top_e_heat = df_filtered["estado"].value_counts().head(10).index
    
    df_heat = df_filtered[
        (df_filtered["nombre_comercial"].isin(top_p_heat)) & 
        (df_filtered["estado"].isin(top_e_heat))
    ]
    
    if not df_heat.empty:
        heatmap_data = pd.crosstab(df_heat["nombre_comercial"], df_heat["estado"])
        
        fig_heat = px.imshow(
            heatmap_data,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Viridis",
            labels=dict(x="Estado", y="Proveedor", color="Num. Quejas")
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Selecciona un rango de fechas o estados más amplio para ver el mapa de calor.")


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














