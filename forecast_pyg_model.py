# Streamlit Dashboard: Proyección Financiera ajustada e interactiva
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Proyección Financiera 2025", layout="wide")
st.title("📊 Proyección de Ventas, Costos, UB, Gastos y EBITDA")

# --- PARÁMETROS GLOBALES ---
archivo_path = "EBITDA_Asimetrix.xlsx"
horizonte_futuro = 12
fecha_max_real = '2025-07-31'

# --- INPUTS INTERACTIVOS ---
crecimiento_input = st.slider("📈 Ajuste de crecimiento sobre predicción (%)", -0.5, 0.5, 0.0, step=0.01)
manual_mode = st.checkbox("⚙️ Usar ajustes manuales en lugar de ratios históricos")

if manual_mode:
    costo_input = st.slider("🧾 % Costos sobre Ventas", 0.0, 1.0, 0.25, step=0.01)
    gasto_input = st.slider("🧾 % Gastos sobre Ventas", 0.0, 1.0, 0.20, step=0.01)

# --- CARGA DE DATOS ---
df = pd.read_excel(archivo_path)
df['Fecha'] = pd.to_datetime(df['Fecha'])
for col in ['Venta', 'Costos', 'Gastos', 'EBITDA', 'Utilidad Bruta']:
    df[col] = df[col].astype(str).str.replace(r'[\$,()]', '', regex=True).str.replace(' ', '')
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=['Venta', 'Costos', 'Gastos'])

df_real = df[df['Fecha'] <= fecha_max_real].copy().sort_values('Fecha')

# --- COMPETENCIA DE MODELOS ---
serie_train = df_real.set_index('Fecha')['Venta']

# SARIMA
sarima_model = SARIMAX(np.log1p(serie_train), order=(1, 1, 1), seasonal_order=(1, 1, 1, 11)).fit(disp=False)
sarima_forecast = np.expm1(sarima_model.forecast(horizonte_futuro))

# Holt-Winters
hw_model = ExponentialSmoothing(serie_train, seasonal='add', seasonal_periods=12).fit()
hw_forecast = hw_model.forecast(horizonte_futuro)

# Prophet
df_prophet = serie_train.reset_index().rename(columns={"Fecha": "ds", "Venta": "y"})
prophet_model = Prophet()
prophet_model.fit(df_prophet)
futuro = prophet_model.make_future_dataframe(periods=horizonte_futuro, freq='MS')
pred_prophet = prophet_model.predict(futuro)
prophet_forecast = pred_prophet.set_index('ds')['yhat'].iloc[-horizonte_futuro:]

# Comparación de métricas
modelos = {
    "SARIMA": sarima_forecast,
    "Holt-Winters": hw_forecast,
    "Prophet": prophet_forecast
}

errores = {}
serie_test = df_real.iloc[int(len(df_real)*0.8):].set_index('Fecha')['Venta']
serie_train = df_real.iloc[:int(len(df_real)*0.8)].set_index('Fecha')['Venta']

for nombre, pred in modelos.items():
    pred = pred[:len(serie_test)]
    mae = mean_absolute_error(serie_test, pred)
    rmse = mean_squared_error(serie_test, pred, squared=False)
    r2 = r2_score(serie_test, pred)
    errores[nombre] = (mae, rmse, r2)

mejor_modelo = min(errores, key=lambda k: errores[k][1])
pred_ventas = modelos[mejor_modelo] * (1 + crecimiento_input)
pred_ventas.index = pd.date_range(start='2025-08-01', periods=horizonte_futuro, freq='MS')

# --- RATIOS ---
if not manual_mode:
    df_real['Costos_pct'] = df_real['Costos'] / df_real['Venta']
    df_real['Gastos_pct'] = df_real['Gastos'] / df_real['Venta']
    costo_pct_50 = df_real['Costos_pct'].mean()
    gasto_pct_50 = df_real['Gastos_pct'].mean()
else:
    costo_pct_50 = costo_input
    gasto_pct_50 = gasto_input

# --- PROYECCIÓN ---
def construir_df(ventas, nombre, c, g):
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

df_proj = construir_df(pred_ventas, 'Base', costo_pct_50, gasto_pct_50)

# --- COMBINAR HISTÓRICO + PROYECCIÓN ---
df_hist = df_real[['Fecha', 'Venta', 'Costos', 'Gastos', 'EBITDA', 'Utilidad Bruta']].copy()
df_hist.columns = ['Fecha', 'Ventas_Hist', 'Costos_Hist', 'Gastos_Hist', 'EBITDA_Hist', 'UB_Hist']
df_hist[['Ventas_Hist', 'Costos_Hist', 'Gastos_Hist', 'EBITDA_Hist', 'UB_Hist']] /= 1000

df_total = pd.concat([df_hist, df_proj], axis=0).sort_values('Fecha')

# --- GRAFICAR ---
metricas = ['Ventas', 'Costos', 'UB', 'Gastos', 'EBITDA']
for metrica in metricas:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df_total['Fecha'], df_total[f'{metrica}_Hist'], label='Histórico', color='black', marker='o')
    ax.plot(df_total['Fecha'], df_total[f'{metrica}_Base'], label='Proyección Base', color='blue', marker='o')
    ax.axvline(pd.to_datetime(fecha_max_real), color='gray', linestyle=':', label='Inicio proyección')
    ax.set_title(f'{metrica} Proyectado')
    ax.set_ylabel('Miles de $')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# --- TABLA YTD ---
mes_corte = '2025-07-31'
df_ytd_24 = df_total[(df_total['Fecha'] >= '2024-01-01') & (df_total['Fecha'] <= mes_corte)]
df_ytd_25_real = df_total[(df_total['Fecha'] >= '2025-01-01') & (df_total['Fecha'] <= fecha_max_real)]
df_ytd_25_proj = df_total[(df_total['Fecha'] > fecha_max_real) & (df_total['Fecha'] <= mes_corte)]

def suma(metrica, df):
    return df[f'{metrica}_Hist'].sum() if f'{metrica}_Hist' in df.columns else 0

def suma_proj(metrica, df):
    return df[f'{metrica}_Base'].sum() if f'{metrica}_Base' in df.columns else 0

def total_ytd25(metrica):
    return suma(metrica, df_ytd_25_real) + suma_proj(metrica, df_ytd_25_proj)

resumen = {
    'Concepto': ['INGRESO NETO', 'TOTAL COSTOS', 'UTILIDAD BRUTA', 'TOTAL GASTOS', 'EBITDA'],
    'YTD 24': [
        suma('Ventas', df_ytd_24), suma('Costos', df_ytd_24), suma('UB', df_ytd_24),
        suma('Gastos', df_ytd_24), suma('EBITDA', df_ytd_24)
    ],
    'YTD 25': [
        total_ytd25('Ventas'), total_ytd25('Costos'), total_ytd25('UB'),
        total_ytd25('Gastos'), total_ytd25('EBITDA')
    ],
    'YTD 25 Ppto': [
        suma('Ventas', df_ytd_25_real), suma('Costos', df_ytd_25_real),
        suma('UB', df_ytd_25_real), suma('Gastos', df_ytd_25_real), suma('EBITDA', df_ytd_25_real)
    ]
}

df_tabla = pd.DataFrame(resumen)
df_tabla['% Cumpl'] = df_tabla['YTD 25'] / df_tabla['YTD 25 Ppto']
df_tabla['% Var.'] = (df_tabla['YTD 25'] / df_tabla['YTD 24']) - 1

# Márgenes
margen_bruto = df_tabla.loc[2, 'YTD 25'] / df_tabla.loc[0, 'YTD 25']
margen_ebitda = df_tabla.loc[4, 'YTD 25'] / df_tabla.loc[0, 'YTD 25']

margen_bruto_24 = df_tabla.loc[2, 'YTD 24'] / df_tabla.loc[0, 'YTD 24']
margen_ebitda_24 = df_tabla.loc[4, 'YTD 24'] / df_tabla.loc[0, 'YTD 24']

df_margenes = pd.DataFrame({
    'Concepto': ['Margen Bruto', 'Margen Ebitda'],
    'YTD 24': [margen_bruto_24, margen_ebitda_24],
    'YTD 25': [margen_bruto, margen_ebitda],
    'YTD 25 Ppto': [None, None],
    '% Cumpl': [None, None],
    '% Var.': [None, None]
})

df_final = pd.concat([df_tabla, df_margenes], ignore_index=True)

st.subheader("📊 Comparativo YTD Jul 25 vs Jul 24 y Presupuesto")
st.dataframe(df_final.style.format({
    'YTD 24': '$ {:,.0f}',
    'YTD 25': '$ {:,.0f}',
    'YTD 25 Ppto': '$ {:,.0f}',
    '% Cumpl': '{:.2%}',
    '% Var.': '{:.2%}',
}), use_container_width=True)

