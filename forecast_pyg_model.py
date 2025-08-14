# Streamlit Dashboard: Proyección Financiera ajustada
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Proyección Financiera 2025", layout="wide")
st.title("📊 Proyección de Ventas, Costos, UB, Gastos y EBITDA")

# --- PARÁMETROS GLOBALES ---
archivo_path = "EBITDA_Asimetrix.xlsx"
horizonte_futuro = 12
fecha_max_real = '2025-07-31'  # Julio 2025
crecimiento_pct = st.slider("🔧 Ajuste de crecimiento sobre predicción (%)", -0.5, 0.5, 0.0, step=0.01)

# --- CARGA DE DATOS ---
df = pd.read_excel(archivo_path)
df['Fecha'] = pd.to_datetime(df['Fecha'])
for col in ['Venta', 'Costos', 'Gastos', 'EBITDA', 'Utilidad Bruta']:
    df[col] = df[col].astype(str).str.replace(r'[\$,()]', '', regex=True).str.replace(' ', '')
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=['Venta', 'Costos', 'Gastos'])

df_real = df[df['Fecha'] <= fecha_max_real].copy().sort_values('Fecha')

# --- ENTRENAMIENTO MODELO ---
serie_train = df_real.set_index('Fecha')['Venta']
modelo_sarima = SARIMAX(np.log1p(serie_train), order=(1, 1, 1), seasonal_order=(1, 1, 1, 11)).fit(disp=False)
fechas_futuras = pd.date_range(start='2025-08-01', periods=horizonte_futuro, freq='MS')
pred_base = np.expm1(modelo_sarima.forecast(horizonte_futuro)) * (1 + crecimiento_pct)
pred_ventas = pd.Series(pred_base.values, index=fechas_futuras)

# --- CÁLCULO DE RATIOS HISTÓRICOS ---
df_real['Costos_pct'] = df_real['Costos'] / df_real['Venta']
df_real['Gastos_pct'] = df_real['Gastos'] / df_real['Venta']

costo_pct_25 = np.percentile(df_real['Costos_pct'], 25)
costo_pct_50 = df_real['Costos_pct'].mean()
costo_pct_75 = np.percentile(df_real['Costos_pct'], 75)

gasto_pct_25 = np.percentile(df_real['Gastos_pct'], 25)
gasto_pct_50 = df_real['Gastos_pct'].mean()
gasto_pct_75 = np.percentile(df_real['Gastos_pct'], 75)

# --- PROYECCIÓN POR ESCENARIO ---
def construir_escenario(ventas, nombre, c, g):
    costos = ventas * c
    gastos = ventas * g
    ub = ventas - costos
    ebitda = ub - gastos
    return pd.DataFrame({
        'Fecha': ventas.index,
        f'Ventas_{nombre}': ventas.values / 1000,
        f'Costos_{nombre}': costos.values / 1000,
        f'UB_{nombre}': ub.values / 1000,
        f'Gastos_{nombre}': gastos.values / 1000,
        f'EBITDA_{nombre}': ebitda.values / 1000
    })

escenarios = {
    'Base': (costo_pct_50, gasto_pct_50),
    'Optimista': (costo_pct_25, gasto_pct_25),
    'Pesimista': (costo_pct_75, gasto_pct_75)
}

proyecciones = []
for esc, (c, g) in escenarios.items():
    df_esc = construir_escenario(pred_ventas, esc, c, g)
    proyecciones.append(df_esc)

# --- DATOS HISTÓRICOS ---
df_hist = df_real[['Fecha', 'Venta', 'Costos', 'Gastos', 'EBITDA', 'Utilidad Bruta']].copy()
df_hist.columns = ['Fecha', 'Ventas_Hist', 'Costos_Hist', 'Gastos_Hist', 'EBITDA_Hist', 'UB_Hist']
df_hist[['Ventas_Hist', 'Costos_Hist', 'Gastos_Hist', 'EBITDA_Hist', 'UB_Hist']] /= 1000

df_total = df_hist.copy()
for df_esc in proyecciones:
    df_total = df_total.merge(df_esc, on='Fecha', how='outer')

# --- GRÁFICOS POR MÉTRICA ---
metricas = ['Ventas', 'Costos', 'UB', 'Gastos', 'EBITDA']
colores = {'Base': 'blue', 'Optimista': 'green', 'Pesimista': 'red'}

for metrica in metricas:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df_total['Fecha'], df_total[f'{metrica}_Hist'], label='Histórico', color='black')
    for esc in escenarios:
        estilo = '--' if esc != 'Base' else '-'
        ax.plot(df_total['Fecha'], df_total[f'{metrica}_{esc}'], label=esc, linestyle=estilo, color=colores[esc])
    ax.axvline(pd.to_datetime(fecha_max_real), color='gray', linestyle=':', label='Inicio proyección')
    ax.set_title(f'{metrica} Proyectado por Escenario')
    ax.set_ylabel('Miles de $')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# --- TABLA COMPARATIVA YTD ---
mes_corte = '2025-07-31'
df_ytd_24 = df_total[(df_total['Fecha'] >= '2024-01-01') & (df_total['Fecha'] <= mes_corte)]
df_ytd_25 = df_total[(df_total['Fecha'] >= '2025-01-01') & (df_total['Fecha'] <= mes_corte)]

resumen = {
    'Concepto': [
        'INGRESO NETO', 'TOTAL COSTOS', 'UTILIDAD BRUTA',
        'TOTAL GASTOS', 'EBITDA', 'Margen Bruto', 'Margen Ebitda'
    ],
    'YTD 24': [
        df_ytd_24['Ventas_Hist'].sum(),
        df_ytd_24['Costos_Hist'].sum(),
        df_ytd_24['UB_Hist'].sum(),
        df_ytd_24['Gastos_Hist'].sum(),
        df_ytd_24['EBITDA_Hist'].sum(),
        df_ytd_24['UB_Hist'].sum() / df_ytd_24['Ventas_Hist'].sum(),
        df_ytd_24['EBITDA_Hist'].sum() / df_ytd_24['Ventas_Hist'].sum()
    ],
    'YTD 25': [
        df_ytd_25['Ventas_Base'].sum(),
        df_ytd_25['Costos_Base'].sum(),
        df_ytd_25['UB_Base'].sum(),
        df_ytd_25['Gastos_Base'].sum(),
        df_ytd_25['EBITDA_Base'].sum(),
        df_ytd_25['UB_Base'].sum() / df_ytd_25['Ventas_Base'].sum(),
        df_ytd_25['EBITDA_Base'].sum() / df_ytd_25['Ventas_Base'].sum()
    ],
    'YTD 25 Ppto': [
        df_ytd_25['Ventas_Hist'].sum(),
        df_ytd_25['Costos_Hist'].sum(),
        df_ytd_25['UB_Hist'].sum(),
        df_ytd_25['Gastos_Hist'].sum(),
        df_ytd_25['EBITDA_Hist'].sum(),
        df_ytd_25['UB_Hist'].sum() / df_ytd_25['Ventas_Hist'].sum(),
        df_ytd_25['EBITDA_Hist'].sum() / df_ytd_25['Ventas_Hist'].sum()
    ]
}

# Calcular cumplimiento y variación
resumen['% Cumpl'] = [
    resumen['YTD 25'][i] / resumen['YTD 25 Ppto'][i] if resumen['YTD 25 Ppto'][i] != 0 else np.nan
    for i in range(len(resumen['YTD 25']))
]
resumen['% Var.'] = [
    (resumen['YTD 25'][i] / resumen['YTD 24'][i] - 1) if resumen['YTD 24'][i] != 0 else np.nan
    for i in range(len(resumen['YTD 24']))
]

df_tabla = pd.DataFrame(resumen)
st.subheader("📊 Comparativo YTD Jul 25 vs Jul 24 y Presupuesto")
st.dataframe(df_tabla.style.format({
    'YTD 24': '$ {:,.0f}',
    'YTD 25': '$ {:,.0f}',
    'YTD 25 Ppto': '$ {:,.0f}',
    '% Cumpl': '{:.2%}',
    '% Var.': '{:.2%}'
}), use_container_width=True)


