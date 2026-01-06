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

geo_mapping = {
    # Nombres comunes en tus datos  :  Nombre exacto en el GeoJSON
    "AGUASCALIENTES": "Aguascalientes",
    "BAJA CALIFORNIA": "Baja California",
    "BAJA CALIFORNIA SUR": "Baja California Sur",
    "CAMPECHE": "Campeche",
    "COAHUILA": "Coahuila de Zaragoza",         # Ojo aquí
    "COLIMA": "Colima",
    "CHIAPAS": "Chiapas",
    "CHIHUAHUA": "Chihuahua",
    "CIUDAD DE MEXICO": "Distrito Federal",     # El GeoJSON usa el nombre antiguo
    "CIUDAD DE MÉXICO": "Distrito Federal",
    "CDMX": "Distrito Federal",
    "DISTRITO FEDERAL": "Distrito Federal",
    "DURANGO": "Durango",
    "GUANAJUATO": "Guanajuato",
    "GUERRERO": "Guerrero",
    "HIDALGO": "Hidalgo",
    "JALISCO": "Jalisco",
    "MEXICO": "México",                         # Estado de México
    "ESTADO DE MEXICO": "México",
    "EDO. DE MEXICO": "México",
    "MICHOACAN": "Michoacán de Ocampo",         # Ojo aquí
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
    "VERACRUZ": "Veracruz de Ignacio de la Llave", # Ojo aquí
    "YUCATAN": "Yucatán",
    "YUCATÁN": "Yucatán",
    "ZACATECAS": "Zacatecas"
}

# --- B. PREPARACIÓN DE COLUMNAS PARA EL MAPA ---

def preparar_estado(texto):
    if texto is None: return None
    # 1. Limpiar básico: Mayúsculas y sin espacios extremos
    t = str(texto).upper().strip()
    # 2. Mapear: Si encuentra la llave usa el valor, si no, deja el texto original
    return geo_mapping.get(t, geo_mapping.get(t.replace("Á","A").replace("É","E").replace("Í","I").replace("Ó","O").replace("Ú","U"), t))

# Creamos columna temporal para el mapa en el DF Principal
df["_estado_mapa"] = df["estado"].apply(preparar_estado)

# Creamos columna temporal en DF Población
col_estado_pob = [c for c in df_pob.columns if "estado" in c.lower() or "entidad" in c.lower()][0]
df_pob["_estado_mapa"] = df_pob[col_estado_pob].apply(preparar_estado)

# Detectamos columna de población (El año más reciente)
cols_anios = [c for c in df_pob.columns if str(c).strip() in ['2025','2024','2023','2022']]
cols_anios.sort(reverse=True) # Orden descendente
col_pob_target = cols_anios[0] if cols_anios else df_pob.select_dtypes('number').columns[-1]

df_pob["_pob_uso"] = df_pob[col_pob_target]

# --- C. PREPARACIÓN DE USUARIOS (PERFIL) ---
if "proveedor_top" not in perfil_df.columns:
    perfil_df = perfil_df.reset_index()
    if "proveedor_top" not in perfil_df.columns:
         perfil_df.rename(columns={perfil_df.columns[0]: "proveedor_top"}, inplace=True)

# Normalizar nombre proveedor para cruces
perfil_df["_prov_join"] = perfil_df["proveedor_top"].astype(str).str.upper().str.strip()
df["_prov_join"] = df["nombre_comercial"].astype(str).str.upper().str.strip()

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
    k1.metric("Total Quejas", f"{total_q:,}")
    k2.metric("Proveedores", f"{df_filtered['nombre_comercial'].nunique()}")
    k3.metric("% Conciliación", f"{pct_concil:.1f}%")
    
    st.markdown("---")

    # =========================================================================
    # SECCIÓN 1: EVOLUCIÓN
    # =========================================================================
    st.subheader("📈 Evolución Temporal")
    
    freq_alias = "M"
    
    df_evo = df_filtered.set_index("fecha_ingreso").groupby(
        [pd.Grouper(freq=freq_alias), "nombre_comercial", "_prov_join"]
    ).size().reset_index(name="conteo")

    col_evo_1, col_evo_2 = st.columns(2)

    with col_evo_1:
        st.markdown("**1. Volumen Absoluto**")
        fig_abs = px.line(
            df_evo, x="fecha_ingreso", y="conteo", color="nombre_comercial", markers=True,
            title="Histórico de Quejas",
            labels={"conteo": "Quejas", "fecha_ingreso": "Fecha"}
        )
        fig_abs.update_layout(legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_abs, use_container_width=True)

    with col_evo_2:
        st.markdown("**2. Tasa Real (x 10k Usuarios)**")
        
        # Cruce con usuarios
        df_evo_rel = pd.merge(
            df_evo, 
            perfil_df[["_prov_join", "usuarios_totales"]], 
            on="_prov_join", 
            how="left"
        )
        
        # Si usuarios es nulo, ponemos 1 para no romper
        df_evo_rel["usuarios_totales"] = pd.to_numeric(df_evo_rel["usuarios_totales"], errors='coerce').fillna(1)
        
        df_evo_rel["tasa"] = (df_evo_rel["conteo"] / df_evo_rel["usuarios_totales"]) * 10000
        
        # Filtro visual para quitar líneas rotas (donde no hubo cruce de usuarios)
        df_plot_rel = df_evo_rel[df_evo_rel["usuarios_totales"] > 100]

        if not df_plot_rel.empty:
            fig_rel = px.line(
                df_plot_rel, x="fecha_ingreso", y="tasa", color="nombre_comercial", markers=True,
                title="Impacto ponderado por usuarios",
                labels={"tasa": "Quejas x 10k Usuarios", "fecha_ingreso": "Fecha"}
            )
            fig_rel.update_layout(legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_rel, use_container_width=True)
        else:
            st.warning("No se pudo calcular la tasa (verifica nombres de proveedores en ambos CSV).")

    # =========================================================================
    # SECCIÓN 2: MAPA GEOGRÁFICO (CON DICCIONARIO CORREGIDO)
    # =========================================================================
    st.subheader("🗺️ Intensidad Geográfica")
    
    try:
        # 1. Agrupar por la llave corregida (_estado_mapa)
        quejas_edo = df_filtered["_estado_mapa"].value_counts().reset_index()
        quejas_edo.columns = ["_estado_mapa", "quejas"]
        
        # 2. Merge con Población
        df_mapa = pd.merge(quejas_edo, df_pob, on="_estado_mapa", how="left")
        
        # 3. Calcular Tasa
        # Aseguramos que sea numérico
        df_mapa["_pob_uso"] = pd.to_numeric(df_mapa["_pob_uso"], errors='coerce').fillna(1)
        df_mapa["tasa_100k"] = (df_mapa["quejas"] / df_mapa["_pob_uso"]) * 100000
        
        fig_map = px.choropleth(
            df_mapa,
            geojson="https://raw.githubusercontent.com/angelnmara/geojson/master/mexico_high.json",
            locations="_estado_mapa",  # Esta columna ahora tiene "Distrito Federal", "Coahuila de Zaragoza", etc.
            featureidkey="properties.name",
            color="tasa_100k",
            color_continuous_scale="Reds",
            title=f"Quejas por 100k hab (Población {col_pob_target})",
            hover_data={"_estado_mapa":True, "quejas":True, "_pob_uso":True}
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig_map, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error técnico en el mapa: {e}")

    # =========================================================================
    # SECCIÓN 3: RESOLUCIÓN Y FOCOS ROJOS
    # =========================================================================
    r2_c1, r2_c2 = st.columns(2)

    with r2_c1:
        st.subheader("📊 Resolución (%)")
        
        df_stack = df_filtered.groupby(["nombre_comercial", "estado_procesal"]).size().reset_index(name="conteo")
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
        st.subheader("🔥 Focos Rojos (Heatmap)")
        
        # --- SOLUCIÓN PARA QUE NO SE VEA SOLO LA CDMX ---
        exclude_cdmx = st.checkbox("Ocultar CDMX/EdoMex para ver mejor el resto", value=True)
        
        # Filtramos datos
        df_heat_source = df_filtered.copy()
        
        if exclude_cdmx:
            # Filtramos usando los nombres del mapa que sabemos que son CDMX
            df_heat_source = df_heat_source[~df_heat_source["_estado_mapa"].isin(["Distrito Federal", "México"])]
            
        top_p = df_heat_source["nombre_comercial"].value_counts().head(10).index
        top_e = df_heat_source["_estado_mapa"].value_counts().head(10).index
        
        df_heat = df_heat_source[
            (df_heat_source["nombre_comercial"].isin(top_p)) &
            (df_heat_source["_estado_mapa"].isin(top_e))
        ]
        
        if not df_heat.empty:
            # Matriz de conteo
            matriz = pd.crosstab(df_heat["nombre_comercial"], df_heat["_estado_mapa"])
            
            # Normalización por población (Opcional, pero recomendada)
            norm_type = st.radio("Métrica:", ["Conteo Directo", "Tasa x 100k hab"], horizontal=True)
            
            if norm_type == "Tasa x 100k hab":
                # Dividir por población
                pob_ref = df_pob.set_index("_estado_mapa")["_pob_uso"]
                pob_subset = pob_ref.reindex(matriz.columns).fillna(1)
                matriz_final = matriz.div(pob_subset, axis=1) * 100000
                fmt = ".1f"
            else:
                matriz_final = matriz
                fmt = "d"
            
            fig_heat = px.imshow(
                matriz_final,
                text_auto=fmt,
                aspect="auto",
                color_continuous_scale="Viridis",
                labels=dict(x="Estado", y="Proveedor", color="Valor"),
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("No hay datos para mostrar con los filtros actuales.")

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
























