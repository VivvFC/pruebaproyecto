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
    
    # --- FILTROS ---
    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            f_min, f_max = df["fecha_ingreso"].min(), df["fecha_ingreso"].max()
            date_range = st.date_input("Periodo", [f_min, f_max])
        
        with c2:
            all_states = sorted(df["estado"].unique())
            sel_states = st.multiselect("Estados", all_states, default=all_states)
            
        with c3:
            all_provs = sorted(df["nombre_comercial"].unique())
            top_5 = df["nombre_comercial"].value_counts().head(5).index.tolist()
            sel_provs = st.multiselect("Proveedores", all_provs, default=top_5)

    # --- APLICAR FILTROS ---
    # Validación de fechas
    if len(date_range) == 2:
        mask = (
            (df["fecha_ingreso"].dt.date >= date_range[0]) & 
            (df["fecha_ingreso"].dt.date <= date_range[1]) &
            (df["estado"].isin(sel_states)) &
            (df["nombre_comercial"].isin(sel_provs))
        )
        df_filtered = df[mask].copy()
    else:
        df_filtered = df.copy()

    # --- KPIs ---
    total_q = len(df_filtered)
    # Cálculo seguro de conciliación
    try:
        if "conciliada" in df_filtered.columns:
            conciliadas = df_filtered["conciliada"].sum()
        else:
            conciliadas = df_filtered["estado_procesal"].astype(str).str.contains("Conciliada|Favor", case=False).sum()
        pct_concil = (conciliadas / total_q * 100) if total_q > 0 else 0
    except:
        pct_concil = 0
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Quejas (Selección)", f"{total_q:,}")
    k2.metric("Proveedores", f"{df_filtered['nombre_comercial'].nunique()}")
    k3.metric("% Conciliación", f"{pct_concil:.1f}%")
    
    st.markdown("---")

    # =========================================================================
    # SECCIÓN 1: EVOLUCIÓN (Absoluta vs Relativa)
    # =========================================================================
    st.subheader("📈 Evolución de Quejas")
    
    # Agrupamos por Mes ('ME' es Month End, el nuevo estándar de Pandas)
    df_evo = df_filtered.set_index("fecha_ingreso").groupby(
        [pd.Grouper(freq="ME"), "nombre_comercial", "nombre_norm"]
    ).size().reset_index(name="conteo")

    col_evo_1, col_evo_2 = st.columns(2)

    with col_evo_1:
        st.markdown("**1. Volumen Total (Quejas)**")
        fig_abs = px.line(
            df_evo, x="fecha_ingreso", y="conteo", color="nombre_comercial", markers=True,
            labels={"conteo": "Quejas", "fecha_ingreso": "Fecha"}
        )
        fig_abs.update_layout(legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_abs, use_container_width=True)

    with col_evo_2:
        st.markdown("**2. Impacto Real (Quejas por cada 10k Usuarios)**")
        
        # Cruzamos con perfil_df para traer 'usuarios_totales'
        # Usamos las columnas normalizadas (nombre_norm y proveedor_norm) para asegurar el cruce
        df_evo_rel = pd.merge(
            df_evo, 
            perfil_df[["proveedor_norm", "usuarios_totales"]], 
            left_on="nombre_norm", 
            right_on="proveedor_norm", 
            how="left"
        )
        
        # Calculamos la tasa
        # Rellenamos nulos con 1 para evitar división por cero si no cruza
        df_evo_rel["tasa"] = (df_evo_rel["conteo"] / df_evo_rel["usuarios_totales"].fillna(1)) * 10000
        
        # Filtramos valores extremos o errores de cruce (donde usuarios sea NaN o 1)
        df_plot_rel = df_evo_rel[df_evo_rel["usuarios_totales"] > 100] # Filtro de seguridad

        if not df_plot_rel.empty:
            fig_rel = px.line(
                df_plot_rel, x="fecha_ingreso", y="tasa", color="nombre_comercial", markers=True,
                labels={"tasa": "Quejas x 10k Usuarios", "fecha_ingreso": "Fecha"}
            )
            fig_rel.update_layout(legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_rel, use_container_width=True)
        else:
            st.warning("No se pudo calcular la tasa. Verifica que los nombres de proveedores coincidan en 'perfil_df'.")

    # =========================================================================
    # SECCIÓN 2: MAPA
    # =========================================================================
    st.subheader("🗺️ Intensidad Geográfica (Tasa)")
    
    c_map, c_debug = st.columns([3, 1])
    
    with c_map:
        try:
            # 1. Agrupar
            quejas_edo = df_filtered["estado_norm"].value_counts().reset_index()
            quejas_edo.columns = ["estado_norm", "quejas"]
            
            # 2. Merge con Población (usando llaves limpias)
            df_mapa = pd.merge(quejas_edo, df_pob, on="estado_norm", how="left")
            
            # 3. Calcular Tasa
            col_pob_tot = [c for c in df_pob.columns if 'total' in c or 'poblacion' in c][0]
            df_mapa["tasa_100k"] = (df_mapa["quejas"] / df_mapa[col_pob_tot]) * 100000
            
            # Nombre real para el tooltip
            col_nom_real = [c for c in df_pob.columns if 'estado' in c or 'entidad' in c][0]

            fig_map = px.choropleth(
                df_mapa,
                geojson="https://raw.githubusercontent.com/angelnmara/geojson/master/mexico_high.json",
                locations="estado_norm",
                featureidkey="properties.name", # Nombres en GeoJSON suelen ser mayúsculas sin acento
                color="tasa_100k",
                color_continuous_scale="Reds",
                hover_name=col_nom_real,
                hover_data=["quejas", col_pob_tot],
                title="Quejas por cada 100k Habitantes"
            )
            fig_map.update_geos(fitbounds="locations", visible=False)
            st.plotly_chart(fig_map, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generando mapa: {e}")
            
    with c_debug:
        with st.expander("¿Problemas con el mapa?"):
            st.write("Verifica que los nombres 'estado_norm' coincidan:")
            st.write("Tus quejas:", df["estado_norm"].unique()[:5])
            st.write("Tu archivo poblacion:", df_pob["estado_norm"].unique()[:5])

    # =========================================================================
    # SECCIÓN 3: COMPOSICIÓN Y RIESGO
    # =========================================================================
    
    r2_c1, r2_c2 = st.columns(2)

    with r2_c1:
        st.subheader("Distribución de Quejas (Sunburst)")
        st.markdown("Haz clic en los segmentos para profundizar.")
        
        # Gráfica Sunburst: Total -> Proveedor -> Estatus
        # Es mucho más limpia que el treemap.
        # Limitamos a Top 10 proveedores para que no sea un caos
        top_10_sun = df_filtered["nombre_comercial"].value_counts().head(10).index
        df_sun = df_filtered[df_filtered["nombre_comercial"].isin(top_10_sun)]
        
        fig_sun = px.sunburst(
            df_sun,
            path=['nombre_comercial', 'estado_procesal'],
            title="Top 10 Proveedores y Resolución"
        )
        st.plotly_chart(fig_sun, use_container_width=True)

    with r2_c2:
        st.subheader("🔥 Matriz de Calor (Riesgo)")
        
        # Top 10 proveedores vs Top 10 estados
        top_p_h = df_filtered["nombre_comercial"].value_counts().head(10).index
        top_e_h = df_filtered["estado"].value_counts().head(10).index
        
        df_heat = df_filtered[
            (df_filtered["nombre_comercial"].isin(top_p_h)) &
            (df_filtered["estado"].isin(top_e_h))
        ]
        
        if not df_heat.empty:
            heatmap_data = pd.crosstab(df_heat["nombre_comercial"], df_heat["estado"])
            fig_heat = px.imshow(
                heatmap_data,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Viridis",
                labels=dict(x="Estado", y="Proveedor", color="Q"),
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("No hay datos suficientes para el mapa de calor.")

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



















