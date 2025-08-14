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
manual_mode = st.checkbox("🛠 Usar ajustes manuales de %Costos/Ventas y %Gastos/Ventas")

if manual_mode:
    costo_input = st.slider("% de Costos sobre Ventas", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    gasto_input = st.slider("% de Gastos sobre Ventas", min_value=0.0, max_value=1.0, value=0.20, step=0.01)

# --- CARGA DE DATOS ---
df = pd.read_excel(archivo_path)
df['Fecha'] = pd.to_datetime(df['Fecha'])
for col in ['Venta', 'Costos', 'Gastos', 'EBITDA', 'Utilidad Bruta']:
    df[col] = df[col].astype(str).str.replace(r'[\$,()]', '', regex=True).str.replace(' ', '')
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=['Venta', 'Costos', 'Gastos'])

# --- DATOS HASTA JULIO 2025 ---
df_real = df[df['Fecha'] <= fecha_max_real].copy().sort_values('Fecha')

# --- MODELO SARIMA ---
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

# --- PROYECCIÓN ---
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

if manual_mode:
    escenarios = {'Base': (costo_input, gasto_input)}
else:
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

# --- COMBINACIÓN TOTAL ---
df_total = df_hist.copy()
for df_esc in proyecciones:
    df_total = df_total.merge(df_esc, on='Fecha', how='outer')

# --- GRAFICAR ---
metricas = ['Ventas', 'Costos', 'UB', 'Gastos', 'EBITDA']
colores = {'Base': 'blue', 'Optimista': 'green', 'Pesimista': 'red'}

for metrica in metricas:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df_total['Fecha'], df_total[f'{metrica}_Hist'], label='Histórico', color='black', marker='o')
    for esc in escenarios:
        estilo = '--' if esc != 'Base' else '-'
        ax.plot(df_total['Fecha'], df_total[f'{metrica}_{esc}'], label=esc, linestyle=estilo, color=colores[esc], marker='o')
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
df_ytd_25_real = df_total[(df_total['Fecha'] >= '2025-01-01') & (df_total['Fecha'] <= fecha_max_real)]
df_ytd_25_forecast = df_total[(df_total['Fecha'] > fecha_max_real) & (df_total['Fecha'] <= mes_corte)]

# Sumar lo real + proyectado
ventas_25 = df_ytd_25_real['Ventas_Hist'].sum() + df_ytd_25_forecast['Ventas_Base'].sum()
costos_25 = df_ytd_25_real['Costos_Hist'].sum() + df_ytd_25_forecast['Costos_Base'].sum()
ub_25     = df_ytd_25_real['UB_Hist'].sum()     + df_ytd_25_forecast['UB_Base'].sum()
gastos_25 = df_ytd_25_real['Gastos_Hist'].sum() + df_ytd_25_forecast['Gastos_Base'].sum()
ebitda_25 = df_ytd_25_real['EBITDA_Hist'].sum() + df_ytd_25_forecast['EBITDA_Base'].sum()

ventas_24 = df_ytd_24['Ventas_Hist'].sum()
costos_24 = df_ytd_24['Costos_Hist'].sum()
ub_24     = df_ytd_24['UB_Hist'].sum()
gastos_24 = df_ytd_24['Gastos_Hist'].sum()
ebitda_24 = df_ytd_24['EBITDA_Hist'].sum()

# Presupuesto asumido igual a histórico (puedes adaptar si cargas otro archivo)
ventas_ppto = df_ytd_25_real['Ventas_Hist'].sum()
costos_ppto = df_ytd_25_real['Costos_Hist'].sum()
ub_ppto     = df_ytd_25_real['UB_Hist'].sum()
gastos_ppto = df_ytd_25_real['Gastos_Hist'].sum()
ebitda_ppto = df_ytd_25_real['EBITDA_Hist'].sum()

margen_bruto_24 = ub_24 / ventas_24 if ventas_24 else 0
margen_bruto_25 = ub_25 / ventas_25 if ventas_25 else 0
margen_ebitda_24 = ebitda_24 / ventas_24 if ventas_24 else 0
margen_ebitda_25 = ebitda_25 / ventas_25 if ventas_25 else 0

# Tabla final
resumen = {
    'Concepto': ['INGRESO NETO', 'TOTAL COSTOS', 'UTILIDAD BRUTA', 'TOTAL GASTOS', 'EBITDA', 'Margen Bruto', 'Margen Ebitda'],
    'YTD 24': [ventas_24, costos_24, ub_24, gastos_24, ebitda_24, margen_bruto_24, margen_ebitda_24],
    'YTD 25': [ventas_25, costos_25, ub_25, gastos_25, ebitda_25, margen_bruto_25, margen_ebitda_25],
    'YTD 25 Ppto': [ventas_ppto, costos_ppto, ub_ppto, gastos_ppto, ebitda_ppto, margen_bruto_25, margen_ebitda_25]
}

df_tabla = pd.DataFrame(resumen)
df_tabla['% Cumpl'] = df_tabla['YTD 25'] / df_tabla['YTD 25 Ppto']
df_tabla['% Var.'] = (df_tabla['YTD 25'] / df_tabla['YTD 24']) - 1

st.subheader("📊 Comparativo YTD Jul 25 vs Jul 24 y Presupuesto")

st.dataframe(
    df_tabla.style.format({
        'YTD 24': '$ {:,.0f}',
        'YTD 25': '$ {:,.0f}',
        'YTD 25 Ppto': '$ {:,.0f}',
        '% Cumpl': '{:.2%}',
        '% Var.': '{:.2%}'
    }).format({
        'YTD 24': '{:,.0f}',
        'YTD 25': '{:,.0f}',
        'YTD 25 Ppto': '{:,.0f}'
    }, subset=['Margen Bruto', 'Margen Ebitda'])
    .format({'Margen Bruto': '{:.2%}', 'Margen Ebitda': '{:.2%}'})
    , use_container_width=True
)
