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

def normalizar_simple(series):
    # Solo quita espacios y pone mayúsculas para cruces internos simples
    if series is None: return None
    return series.astype(str).str.upper().str.strip()

# --- B. DICCIONARIO DE CORRECCIÓN PARA GEOJSON (ESTANDARIZACIÓN) ---
# El mapa de México suele requerir nombres muy específicos. 
# Mapeamos tus nombres a los del GeoJSON estándar.
map_estado_geojson = {
    "AGUASCALIENTES": "Aguascalientes",
    "BAJA CALIFORNIA": "Baja California",
    "BAJA CALIFORNIA SUR": "Baja California Sur",
    "CAMPECHE": "Campeche",
    "COAHUILA": "Coahuila de Zaragoza",
    "COAHUILA DE ZARAGOZA": "Coahuila de Zaragoza",
    "COLIMA": "Colima",
    "CHIAPAS": "Chiapas",
    "CHIHUAHUA": "Chihuahua",
    "CIUDAD DE MEXICO": "Distrito Federal", # Muchos GeoJSON viejos usan DF
    "CIUDAD DE MÉXICO": "Distrito Federal",
    "CDMX": "Distrito Federal",
    "DISTRITO FEDERAL": "Distrito Federal",
    "DURANGO": "Durango",
    "GUANAJUATO": "Guanajuato",
    "GUERRERO": "Guerrero",
    "HIDALGO": "Hidalgo",
    "JALISCO": "Jalisco",
    "MEXICO": "México", # Estado de México
    "ESTADO DE MEXICO": "México",
    "MICHOACAN": "Michoacán de Ocampo",
    "MICHOACÁN": "Michoacán de Ocampo",
    "MORELOS": "Morelos",
    "NAYARIT": "Nayarit",
    "NUEVO LEON": "Nuevo León",
    "NUEVO LEÓN": "Nuevo León",
    "OAXACA": "Oaxaca",
    "PUEBLA": "Puebla",
    "QUERETARO": "Querétaro",
    "QUERÉTARO": "Querétaro",
    "QUINTANA ROO": "Quintana Roo",
    "SAN LUIS POTOSI": "San Luis Potosí",
    "SAN LUIS POTOSÍ": "San Luis Potosí",
    "SINALOA": "Sinaloa",
    "SONORA": "Sonora",
    "TABASCO": "Tabasco",
    "TAMAULIPAS": "Tamaulipas",
    "TLAXCALA": "Tlaxcala",
    "VERACRUZ": "Veracruz de Ignacio de la Llave",
    "VERACRUZ DE IGNACIO DE LA LLAVE": "Veracruz de Ignacio de la Llave",
    "YUCATAN": "Yucatán",
    "YUCATÁN": "Yucatán",
    "ZACATECAS": "Zacatecas"
}

# --- C. PREPARACIÓN DF PRINCIPAL ---
# 1. Normalizamos a mayúsculas para buscar en el diccionario
df["_temp_upper"] = normalizar_simple(df["estado"])
# 2. Creamos la columna 'estado_mapa' que SI coincide con el GeoJSON
df["estado_mapa"] = df["_temp_upper"].map(map_estado_geojson).fillna(df["estado"])

# Llave de proveedor normalizada para cruces
df["_key_prov"] = normalizar_simple(df["nombre_comercial"])

# --- D. PREPARACIÓN DF POBLACIÓN ---
# Detectamos columna estado
col_estado_pob = [c for c in df_pob.columns if "estado" in c.lower() or "entidad" in c.lower()][0]
# Normalizamos y Mapeamos
df_pob["_temp_upper"] = normalizar_simple(df_pob[col_estado_pob])
df_pob["estado_mapa"] = df_pob["_temp_upper"].map(map_estado_geojson).fillna(df_pob[col_estado_pob])

# Detectamos columna de población (Año más reciente)
cols_num = [c for c in df_pob.columns if str(c).strip() in ['2024','2023','2022','2021','2020']]
cols_num.sort(reverse=True)
col_pob_target = cols_num[0] if cols_num else df_pob.select_dtypes('number').columns[-1]

df_pob["_poblacion_uso"] = df_pob[col_pob_target]

# --- E. PREPARACIÓN PERFIL (USUARIOS) ---
if "proveedor_top" not in perfil_df.columns:
    perfil_df = perfil_df.reset_index()
    if "proveedor_top" not in perfil_df.columns:
         perfil_df.rename(columns={perfil_df.columns[0]: "proveedor_top"}, inplace=True)

perfil_df["_key_prov"] = normalizar_simple(perfil_df["proveedor_top"])

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
    try:
        if "conciliada" in df_filtered.columns:
            conciliadas = df_filtered["conciliada"].sum()
        else:
            conciliadas = df_filtered["estado_procesal"].astype(str).str.contains("Conciliada|Favor", case=False).sum()
        pct_concil = (conciliadas / total_q * 100) if total_q > 0 else 0
    except:
        pct_concil = 0
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Quejas Totales", f"{total_q:,}")
    k2.metric("Proveedores", f"{df_filtered['nombre_comercial'].nunique()}")
    k3.metric("% Conciliación", f"{pct_concil:.1f}%")
    
    st.markdown("---")

    # =========================================================================
    # SECCIÓN 1: EVOLUCIÓN
    # =========================================================================
    st.subheader("📈 Evolución Temporal")
    
    freq_alias = "M" # Usamos 'M' para máxima compatibilidad
    
    df_evo = df_filtered.set_index("fecha_ingreso").groupby(
        [pd.Grouper(freq=freq_alias), "nombre_comercial", "_key_prov"]
    ).size().reset_index(name="conteo")

    col_evo_1, col_evo_2 = st.columns(2)

    with col_evo_1:
        st.markdown("**1. Volumen Total**")
        fig_abs = px.line(
            df_evo, x="fecha_ingreso", y="conteo", color="nombre_comercial", markers=True,
            title="Quejas Mensuales",
            labels={"conteo": "Quejas", "fecha_ingreso": "Fecha", "nombre_comercial": "Proveedor"}
        )
        fig_abs.update_layout(legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_abs, use_container_width=True)

    with col_evo_2:
        st.markdown("**2. Tasa Real (x 10k Usuarios)**")
        # Merge con usuarios
        df_evo_rel = pd.merge(
            df_evo, 
            perfil_df[["_key_prov", "usuarios_totales"]], 
            on="_key_prov", 
            how="left"
        )
        # Cálculo Tasa
        df_evo_rel["tasa"] = (df_evo_rel["conteo"] / df_evo_rel["usuarios_totales"].fillna(1)) * 10000
        df_plot_rel = df_evo_rel[df_evo_rel["usuarios_totales"] > 100]

        if not df_plot_rel.empty:
            fig_rel = px.line(
                df_plot_rel, x="fecha_ingreso", y="tasa", color="nombre_comercial", markers=True,
                title="Impacto Normalizado por Usuarios",
                labels={"tasa": "Quejas x 10k Usuarios", "fecha_ingreso": "Fecha"}
            )
            fig_rel.update_layout(legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_rel, use_container_width=True)
        else:
            st.info("No hay datos de usuarios suficientes.")

    # =========================================================================
    # SECCIÓN 2: MAPA GEOGRÁFICO (CORREGIDO)
    # =========================================================================
    st.subheader("🗺️ Mapa de Intensidad (Normalizado)")
    
    try:
        # 1. Agrupar por 'estado_mapa' (el nombre corregido para GeoJSON)
        quejas_edo = df_filtered["estado_mapa"].value_counts().reset_index()
        quejas_edo.columns = ["estado_mapa", "quejas"]
        
        # 2. Merge con Población usando la misma llave 'estado_mapa'
        df_mapa = pd.merge(quejas_edo, df_pob, on="estado_mapa", how="left")
        
        # 3. Tasa Normalizada
        df_mapa["tasa_100k"] = (df_mapa["quejas"] / df_mapa["_poblacion_uso"]) * 100000
        
        fig_map = px.choropleth(
            df_mapa,
            geojson="https://raw.githubusercontent.com/angelnmara/geojson/master/mexico_high.json",
            locations="estado_mapa",
            featureidkey="properties.name",
            color="tasa_100k",
            color_continuous_scale="Reds",
            hover_name="estado_mapa",
            hover_data={"estado_mapa":False, "quejas":True, "_poblacion_uso":True},
            title="Quejas por cada 100,000 Habitantes"
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig_map, use_container_width=True)
    except Exception as e:
        st.error(f"Error en mapa: {e}")

    # =========================================================================
    # SECCIÓN 3: RESOLUCIÓN Y FOCOS ROJOS (NORMALIZADO)
    # =========================================================================
    r2_c1, r2_c2 = st.columns(2)

    with r2_c1:
        st.subheader("📊 Resolución de Quejas")
        # Barras 100%
        df_stack = df_filtered.groupby(["nombre_comercial", "estado_procesal"]).size().reset_index(name="conteo")
        
        # Cálculo manual de %
        totals = df_stack.groupby("nombre_comercial")["conteo"].transform("sum")
        df_stack["porcentaje"] = (df_stack["conteo"] / totals) * 100
        
        order_prov = df_filtered["nombre_comercial"].value_counts().index
        
        fig_stack = px.bar(
            df_stack,
            y="nombre_comercial",
            x="porcentaje",
            color="estado_procesal",
            orientation='h',
            text_auto=".0f",
            category_orders={"nombre_comercial": order_prov}
        )
        fig_stack.update_layout(barmode='stack', legend=dict(orientation="h", y=-0.3), xaxis_title="% Total", yaxis_title=None)
        st.plotly_chart(fig_stack, use_container_width=True)

    with r2_c2:
        st.subheader("🔥 Focos Rojos (Tasa x 100k Hab)")
        st.markdown("Normalizado por población para eliminar sesgo CDMX.")
        
        # 1. Filtramos Top Proveedores y Estados
        top_p = df_filtered["nombre_comercial"].value_counts().head(10).index
        top_e = df_filtered["estado_mapa"].value_counts().head(10).index
        
        df_heat = df_filtered[
            (df_filtered["nombre_comercial"].isin(top_p)) &
            (df_filtered["estado_mapa"].isin(top_e))
        ]
        
        if not df_heat.empty:
            # 2. Creamos matriz de conteo (Crosstab)
            # Usamos 'estado_mapa' para poder cruzar con población fácilmente
            matriz_conteo = pd.crosstab(df_heat["nombre_comercial"], df_heat["estado_mapa"])
            
            # 3. NORMALIZACIÓN: Dividir cada columna (estado) por su población
            # Obtenemos serie de población indexada por estado_mapa
            pob_series = df_pob.set_index("estado_mapa")["_poblacion_uso"]
            
            # Alineamos las series (solo los estados que están en la matriz)
            # Reindexamos pob_series para que coincida con las columnas de la matriz
            pob_subset = pob_series.reindex(matriz_conteo.columns).fillna(1) # fillna 1 por seguridad
            
            # Dividimos y multiplicamos por 100k
            matriz_tasa = matriz_conteo.div(pob_subset, axis=1) * 100000
            
            fig_heat = px.imshow(
                matriz_tasa,
                text_auto=".1f", # Mostrar 1 decimal
                aspect="auto",
                color_continuous_scale="Viridis", # Escala de colores
                labels=dict(x="Estado", y="Proveedor", color="Tasa (x100k)"),
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Datos insuficientes.")

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























