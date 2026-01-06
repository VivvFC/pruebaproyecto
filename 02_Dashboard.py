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
# PESTAÑA 1: ANÁLISIS DESCRIPTIVO
# =============================================================================
with tab1:
    st.markdown("### 📊 Panorama General de Reclamaciones")
    st.markdown("Visión estratégica del volumen, montos y tiempos de resolución de las quejas ante PROFECO.")
    st.markdown("---")

    # -------------------------------------------------------------------------
    # 1. SECCIÓN DE KPIs (INDICADORES CLAVE)
    # -------------------------------------------------------------------------
    # Calculamos métricas rápidas sobre los datos filtrados
    total_quejas = len(df_filtered)
    monto_total = df_filtered['monto_reclamado'].sum()
    
    # Conciliación: Porcentaje de casos que terminaron en "Conciliada"
    conciliados = df_filtered[df_filtered['estado_procesal'] == 'Conciliada'].shape[0]
    tasa_conciliacion = (conciliados / total_quejas * 100) if total_quejas > 0 else 0
    
    # Tiempo promedio (si calculaste la columna 'dias_resolucion' en el tratamiento)
    if 'dias_resolucion' in df_filtered.columns:
        tiempo_promedio = df_filtered['dias_resolucion'].mean()
    else:
        tiempo_promedio = 0

    # Desplegamos 4 métricas en columnas
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    kpi1.metric(
        label="📂 Total Expedientes", 
        value=f"{total_quejas:,.0f}",
        help="Número total de quejas en el periodo seleccionado."
    )
    
    kpi2.metric(
        label="💰 Monto Reclamado", 
        value=f"${monto_total/1000000:.1f}M", 
        delta="MXN",
        help="Suma total de los montos reclamados (Millones de Pesos)."
    )
    
    kpi3.metric(
        label="🤝 Tasa de Conciliación", 
        value=f"{tasa_conciliacion:.1f}%",
        help="Porcentaje de expedientes que lograron conciliarse."
    )
    
    kpi4.metric(
        label="⏱️ Tiempo Prom. Resolución", 
        value=f"{tiempo_promedio:.0f} días",
        help="Días promedio entre fecha de ingreso y cierre."
    )
    
    st.markdown("---")

    # -------------------------------------------------------------------------
    # 2. FILA 1: TENDENCIAS Y VOLUMEN (GRÁFICAS PRINCIPALES)
    # -------------------------------------------------------------------------
    col_main_1, col_main_2 = st.columns([2, 1])  # Columna izquierda más ancha

    with col_main_1:
        st.subheader("📈 Evolución Temporal de Quejas")
        # Agrupamos por mes para ver la tendencia limpia
        # Aseguramos que fecha_ingreso sea datetime
        df_trend = df_filtered.set_index('fecha_ingreso').resample('M').size().reset_index(name='Quejas')
        
        fig_trend = px.line(
            df_trend, 
            x='fecha_ingreso', 
            y='Quejas', 
            markers=True,
            title="Tendencia Mensual de Expedientes Ingresados",
            color_discrete_sequence=['#2E86C1'] # Azul corporativo
        )
        fig_trend.update_layout(xaxis_title="Fecha", yaxis_title="Número de Quejas", hovermode="x unified")
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_main_2:
        st.subheader("🏆 Top 10 Proveedores")
        # Contamos y ordenamos
        top_prov = df_filtered['proveedor_top'].value_counts().head(10).reset_index()
        top_prov.columns = ['Proveedor', 'Quejas']
        
        fig_bar = px.bar(
            top_prov, 
            x='Quejas', 
            y='Proveedor', 
            orientation='h',
            text='Quejas',
            color='Quejas',
            color_continuous_scale='Blues',
            title="Proveedores con más Reclamaciones"
        )
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # -------------------------------------------------------------------------
    # 3. FILA 2: ANÁLISIS DE CAUSAS Y MONTOS (SUNBURST Y BOXPLOT)
    # -------------------------------------------------------------------------
    st.markdown("#### 🔍 Profundización en Causas y Costos")
    col_deep_1, col_deep_2 = st.columns(2)

    with col_deep_1:
        st.markdown("**Distribución Jerárquica: ¿De qué se quejan?**")
        # Gráfico Sunburst: Categoría -> Motivo
        # Usamos head(15) en motivos para no saturar visualmente
        fig_sun = px.sunburst(
            df_filtered, 
            path=['categoria_problema', 'motivo_reclamacion'], 
            values='costo_bien_servicio', # Tamaño por costo del servicio (o usa 'count' si prefieres frecuencia)
            color='categoria_problema',
            color_discrete_sequence=px.colors.qualitative.Pastel,
            title="Jerarquía de Problemas (Click para explorar)"
        )
        fig_sun.update_layout(margin=dict(t=30, l=0, r=0, b=0))
        st.plotly_chart(fig_sun, use_container_width=True)

    with col_deep_2:
        st.markdown("**Dispersión de Montos Reclamados**")
        # Checkbox para interactividad en escala
        log_scale = st.checkbox("Usar Escala Logarítmica (Recomendado para ver Outliers)", value=True, key="log_tab1")
        
        # Filtramos ceros para evitar errores en log
        df_money = df_filtered[df_filtered['monto_reclamado'] > 0]
        
        fig_box = px.box(
            df_money, 
            x='proveedor_top', 
            y='monto_reclamado',
            color='proveedor_top',
            log_y=log_scale,
            title=f"Distribución de Montos por Proveedor ({'Log' if log_scale else 'Lineal'})",
            points="outliers" # Solo mostramos outliers como puntos
        )
        fig_box.update_layout(showlegend=False, yaxis_title="Monto ($MXN)")
        st.plotly_chart(fig_box, use_container_width=True)

    # -------------------------------------------------------------------------
    # 4. FILA 3: MAPA DE CALOR (CORRELACIÓN) - ¡EL FACTOR WOW!
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🔥 Mapa de Calor: ¿Dónde le duele a cada empresa?")
    st.markdown("Identifica patrones visuales: ¿Qué categoría afecta más a cada proveedor?")

    # Creamos la matriz cruzada
    heatmap_data = pd.crosstab(df_filtered['proveedor_top'], df_filtered['categoria_problema'])
    
    fig_heat = px.imshow(
        heatmap_data,
        text_auto=True, # Muestra los números en las celdas
        aspect="auto",
        color_continuous_scale="Reds",
        labels=dict(x="Categoría del Problema", y="Proveedor", color="No. Quejas"),
        title="Intensidad de Quejas por Categoría y Proveedor"
    )
    fig_heat.update_xaxes(side="top") # Pone las etiquetas arriba para leer mejor
    st.plotly_chart(fig_heat, use_container_width=True)

    # -------------------------------------------------------------------------
    # 5. EXPANDER CON DATOS RAW (PARA TRANSPARENCIA)
    # -------------------------------------------------------------------------
    with st.expander("📂 Ver Detalle de los Datos Filtrados"):
        st.dataframe(
            df_filtered[['fecha_ingreso', 'proveedor', 'motivo_reclamacion', 
                         'monto_reclamado', 'costo_bien_servicio', 'estado_procesal']]
            .sort_values(by='fecha_ingreso', ascending=False)
            .head(1000) # Limitamos a 1000 para rendimiento
        )
        st.caption("Mostrando los últimos 1000 registros filtrados.")

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






