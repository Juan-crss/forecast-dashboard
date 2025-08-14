# Streamlit Dashboard: Predicción de Ventas y Simulación de P&G Interactiva (con carga directa)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

# --- FUNCIONES AUXILIARES ---
def calcular_metricas(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, mape, r2

def proyectar_estado(ventas, nombre, costo_pct, gasto_pct):
    costos = ventas * costo_pct
    gastos = ventas * gasto_pct
    ub = ventas - costos
    ebitda = ub - gastos
    return pd.DataFrame({
        'Fecha': ventas.index,
        f'Ingresos_{nombre}': ventas.values,
        f'Costos_{nombre}': costos.values,
        f'Gastos_{nombre}': gastos.values,
        f'UB_{nombre}': ub.values,
        f'EBITDA_{nombre}': ebitda.values
    })

# --- DASHBOARD ---
st.set_page_config(page_title="Predicción de Ventas y P&G", layout="wide")
st.title("📈 Predicción de Ventas y Simulación de Estado de Resultados")

# --- PARÁMETROS ---
horizonte_futuro = 12
archivo_path = r"C:\\Users\\Juanr\Downloads\\EBITDA_AfA_USA.xlsx"

# --- CARGA DIRECTA ---
df = pd.read_excel(archivo_path)
df['Fecha'] = pd.to_datetime(df['Fecha'])
for col in ['Venta', 'Costos', 'Gastos']:
    df[col] = df[col].astype(str).str.replace(r'[\$,()]', '', regex=True).str.replace(' ', '')
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=['Venta', 'Costos', 'Gastos'])
df_real = df.sort_values('Fecha')

# --- SPLIT ---
col_fecha = 'Fecha'
col_venta = 'Venta'
n = len(df_real)
n_train = int(n * 0.8)
serie_train = df_real.iloc[:n_train][col_venta]
serie_test = df_real.iloc[n_train:][col_venta]
fechas_test = df_real.iloc[n_train:][col_fecha]
resultados = []
predicciones_futuras = {}

# --- ETS ---
modelo_ets = ExponentialSmoothing(serie_train, trend='add', seasonal='add', seasonal_periods=12).fit()
pred_test_ets = modelo_ets.forecast(len(serie_test))
mae_v, rmse_v, mape_v, r2_v = calcular_metricas(serie_test, pred_test_ets)
resultados.append({'Modelo': 'ETS', 'MAPE': mape_v})
predicciones_futuras['ETS'] = modelo_ets.forecast(horizonte_futuro)

# --- SARIMA ---
serie_train_log = np.log1p(serie_train)
modelo_sarima = SARIMAX(serie_train_log, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
pred_sarima_futuro = modelo_sarima.forecast(horizonte_futuro)
predicciones_futuras['SARIMA'] = np.expm1(pred_sarima_futuro)

# --- Prophet ---
df_prophet = df_real[['Fecha', 'Venta']].rename(columns={'Fecha': 'ds', 'Venta': 'y'})
modelo_prophet = Prophet()
modelo_prophet.fit(df_prophet)
future = modelo_prophet.make_future_dataframe(periods=horizonte_futuro, freq='MS')
forecast = modelo_prophet.predict(future)
pred_prophet = forecast.set_index('ds')['yhat'][-horizonte_futuro:]
predicciones_futuras['Prophet'] = pred_prophet

# --- Interfaz ---
modelo_opcion = st.selectbox("Selecciona el modelo para proyectar ventas:", list(predicciones_futuras.keys()))
pred_final = predicciones_futuras[modelo_opcion]
crecimiento_input = st.slider("Ajuste de crecimiento sobre predicción (%)", -0.5, 0.5, 0.0, step=0.01)
ventas_ajustadas = pred_final * (1 + crecimiento_input)
manual_mode = st.checkbox("Usar ajustes manuales en lugar de ratios históricos")

if not manual_mode:
    df_real['Costos_pct'] = df_real['Costos'] / df_real['Venta']
    df_real['Gastos_pct'] = df_real['Gastos'] / df_real['Venta']
    costo_pct_25 = np.percentile(df_real['Costos_pct'], 25)
    costo_pct_50 = df_real['Costos_pct'].mean()
    costo_pct_75 = np.percentile(df_real['Costos_pct'], 75)
    gasto_pct_25 = np.percentile(df_real['Gastos_pct'], 25)
    gasto_pct_50 = df_real['Gastos_pct'].mean()
    gasto_pct_75 = np.percentile(df_real['Gastos_pct'], 75)
    df_base = proyectar_estado(ventas_ajustadas, 'Base', costo_pct_50, gasto_pct_50)
    df_opt  = proyectar_estado(ventas_ajustadas, 'Optimista', costo_pct_25, gasto_pct_25)
    df_pes  = proyectar_estado(ventas_ajustadas, 'Pesimista', costo_pct_75, gasto_pct_75)
    df_financiero = df_base.merge(df_opt, on='Fecha').merge(df_pes, on='Fecha')
    st.subheader("📊 Escenarios de Estado de Resultados")
    st.line_chart(df_financiero.set_index('Fecha')[["EBITDA_Base", "EBITDA_Optimista", "EBITDA_Pesimista"]])
    st.dataframe(df_financiero, use_container_width=True)
    st.download_button("📥 Descargar escenarios como Excel", data=df_financiero.to_csv(index=False).encode('utf-8'), file_name="escenarios_financieros.csv", mime="text/csv")
else:
    costo_input = st.slider("% de Costos sobre Ventas", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    gasto_input = st.slider("% de Gastos sobre Ventas", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
    df_pyg = proyectar_estado(ventas_ajustadas, 'Escenario', costo_input, gasto_input)
    df_pyg.set_index('Fecha', inplace=True)
    st.subheader("📊 Ventas Proyectadas y Resultados Financieros")
    st.line_chart(df_pyg[[f'Ingresos_Escenario', f'EBITDA_Escenario']])
    st.dataframe(df_pyg, use_container_width=True)
    st.download_button("📥 Descargar resultados como Excel", data=df_pyg.to_csv(index=True).encode('utf-8'), file_name="proyeccion_resultados.csv", mime="text/csv")
