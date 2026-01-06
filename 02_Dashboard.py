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
    
    # --- BARRA DE FILTROS SUPERIOR ---
    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            # Filtro Fechas
            f_min, f_max = df["fecha_ingreso"].min(), df["fecha_ingreso"].max()
            date_range = st.date_input("Periodo de Análisis", [f_min, f_max])
        
        with c2:
            # Filtro Estados
            all_states = sorted(df["estado"].unique())
            sel_states = st.multiselect("Filtrar Estados", all_states, default=all_states)
            
        with c3:
            # Filtro Proveedores (Multiselect para las gráficas de evolución)
            all_provs = sorted(df["nombre_comercial"].unique())
            # Pre-seleccionamos el Top 5 para no saturar al inicio
            top_5 = df["nombre_comercial"].value_counts().head(5).index.tolist()
            sel_provs = st.multiselect("Filtrar Proveedores", all_provs, default=top_5)

    # --- APLICACIÓN DE FILTROS ---
    if len(date_range) == 2:
        mask = (
            (df["fecha_ingreso"].dt.date >= date_range[0]) & 
            (df["fecha_ingreso"].dt.date <= date_range[1]) &
            (df["estado"].isin(sel_states)) &
            (df["nombre_comercial"].isin(sel_provs))
        )
        df_filtered = df[mask].copy()
    else:
        df_filtered = df.copy() # Fallback si no hay fechas seleccionadas

    # --- KPIs ---
    total_q = len(df_filtered)
    
    # Cálculo de Conciliación (Buscando texto si no hay columna numérica)
    if "conciliada" in df_filtered.columns:
        conciliadas = df_filtered["conciliada"].sum()
    else:
        # Busca "Conciliada" o "Favor" en el estatus
        conciliadas = df_filtered["estado_procesal"].astype(str).str.contains("Conciliada|Favor", case=False).sum()
        
    pct_concil = (conciliadas / total_q * 100) if total_q > 0 else 0
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Quejas (Selección)", f"{total_q:,}")
    k2.metric("Proveedores Analizados", f"{df_filtered['nombre_comercial'].nunique()}")
    k3.metric("Tasa de Conciliación", f"{pct_concil:.1f}%")
    
    st.markdown("---")

    # =========================================================================
    # SECCIÓN 1: EVOLUCIÓN COMPARATIVA (Absoluta vs Relativa)
    # =========================================================================
    st.subheader("📈 Evolución de Quejas")
    
    # Preparamos datos base agrupados por mes
    df_evo = df_filtered.set_index("fecha_ingreso").groupby(
        [pd.Grouper(freq="M"), "nombre_comercial"]
    ).size().reset_index(name="conteo")

    col_evo_1, col_evo_2 = st.columns(2)

    with col_evo_1:
        st.markdown("**1. Volumen Total (Número de Quejas)**")
        st.caption("Muestra quién tiene más quejas en números brutos.")
        fig_abs = px.line(
            df_evo, 
            x="fecha_ingreso", 
            y="conteo", 
            color="nombre_comercial",
            markers=True,
            labels={"conteo": "Quejas", "fecha_ingreso": "Mes"}
        )
        fig_abs.update_layout(legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_abs, use_container_width=True)

    with col_evo_2:
        st.markdown("**2. Impacto Real (Quejas por cada 10k Usuarios)**")
        st.caption("Divide las quejas entre el total de usuarios de cada empresa.")
        
        if col_usuarios_real:
            # Cruzamos con perfil_df para obtener usuarios
            # Asumimos que el índice de perfil_df es el nombre del proveedor
            
            # Función auxiliar para buscar usuarios en perfil_df
            def get_users(prov_name):
                try:
                    # Intenta buscar directo en el index
                    if prov_name in perfil_df.index:
                        return perfil_df.loc[prov_name, col_usuarios_real]
                    # Si no, busca si hay columna de nombre
                    elif "proveedor" in perfil_df.columns:
                        val = perfil_df.loc[perfil_df["proveedor"] == prov_name, col_usuarios_real]
                        return val.iloc[0] if not val.empty else np.nan
                    return np.nan
                except:
                    return np.nan

            df_evo["usuarios"] = df_evo["nombre_comercial"].apply(get_users)
            
            # Calculamos tasa (Quejas por cada 10,000 usuarios)
            df_evo["tasa"] = (df_evo["conteo"] / df_evo["usuarios"]) * 10000
            
            fig_rel = px.line(
                df_evo.dropna(subset=["tasa"]), 
                x="fecha_ingreso", 
                y="tasa", 
                color="nombre_comercial",
                markers=True,
                labels={"tasa": "Quejas x 10k Usuarios", "fecha_ingreso": "Mes"}
            )
            fig_rel.update_layout(legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_rel, use_container_width=True)
        else:
            st.warning("⚠️ No se encontró una columna de 'Usuarios' en perfil_df. Verifica el nombre del CSV.")

    # =========================================================================
    # SECCIÓN 2: MAPA Y EFECTIVIDAD DE RESOLUCIÓN
    # =========================================================================
    
    r2_c1, r2_c2 = st.columns([1, 1])

    with r2_c1:
        st.subheader("🗺️ Intensidad Geográfica (Tasa)")
        # 1. Agrupar datos actuales por estado (usando nombre limpio)
        quejas_edo = df_filtered["estado_clean"].value_counts().reset_index()
        quejas_edo.columns = ["estado_clean", "quejas"]
        
        # 2. Merge con población (usando nombre limpio)
        df_mapa = pd.merge(quejas_edo, df_pob, on="estado_clean", how="left")
        
        # 3. Detectar columna de población total
        col_pob_total = [c for c in df_pob.columns if 'total' in c or 'poblacion' in c][0]
        
        # 4. Calcular tasa por 100k habitantes
        df_mapa["tasa_100k"] = (df_mapa["quejas"] / df_mapa[col_pob_total]) * 100000
        
        # Recuperamos el nombre bonito del estado para el tooltip
        col_nombre_orig = [c for c in df_pob.columns if 'estado' in c or 'entidad' in c][0]
        
        fig_map = px.choropleth(
            df_mapa,
            geojson="https://raw.githubusercontent.com/angelnmara/geojson/master/mexico_high.json",
            locations="estado_clean", # Llave de cruce limpia
            featureidkey="properties.name", # El geojson suele estar en mayúsculas/sin acentos
            color="tasa_100k",
            color_continuous_scale="Reds",
            hover_name=col_nombre_orig, # Mostrar nombre original
            hover_data=["quejas", col_pob_total],
            title="Quejas por cada 100k Habitantes"
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig_map, use_container_width=True)

    with r2_c2:
        st.subheader("📊 Efectividad de Resolución")
        st.markdown("¿Cómo termina cada proveedor sus quejas? (Proporción)")
        
        # Gráfica de Barras Apiladas al 100%
        # Esto reemplaza al Treemap confuso. Muestra claramente el % de conciliación.
        df_stack = df_filtered.groupby(["nombre_comercial", "estado_procesal"]).size().reset_index(name="conteo")
        
        fig_stack = px.bar(
            df_stack,
            x="nombre_comercial",
            y="conteo",
            color="estado_procesal", # Colores por estatus (Conciliada, Trámite, etc.)
            title="Distribución de Estatus por Proveedor",
            barmode="stack",
            barnorm="percent", # Normaliza al 100%
            text_auto=".1f"    # Muestra porcentaje
        )
        fig_stack.update_layout(yaxis_title="% del Total de Quejas")
        st.plotly_chart(fig_stack, use_container_width=True)

    # =========================================================================
    # SECCIÓN 3: FOCOS ROJOS (HEATMAP)
    # =========================================================================
    st.subheader("🔥 Matriz de Riesgo: Proveedor vs Estado")
    
    # Filtramos Top 15 estados y proveedores para que se lea bien
    top_p_h = df_filtered["nombre_comercial"].value_counts().head(12).index
    top_e_h = df_filtered["estado"].value_counts().head(12).index
    
    df_heat = df_filtered[
        (df_filtered["nombre_comercial"].isin(top_p_h)) &
        (df_filtered["estado"].isin(top_e_h))
    ]
    
    heatmap_data = pd.crosstab(df_heat["nombre_comercial"], df_heat["estado"])
    
    fig_heat = px.imshow(
        heatmap_data,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Viridis",
        labels=dict(x="Estado", y="Proveedor", color="Quejas"),
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

















