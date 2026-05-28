"""
Mercado Justo — Bot de Telegram para productores de café.

Comandos:
  /precio        → Precio actual del café (en vivo)
  /pronostico    → Pronóstico ARIMA a 3, 6 y 12 meses
  /recomendacion → VENDER o MANTENER con justificación
  /senal         → Probabilidad de alza (Random Forest)
  /brecha        → Brecha productor vs mercado internacional
  /cultivo       → Plan óptimo de cultivo (PuLP)
  /calcular      → Calculadora personalizada por estado/municipio
  /resumen       → Todo junto en un mensaje
  /ayuda         → Lista de comandos

Uso:
  cp config.py.example config.py   # poner tu token
  bash setup.sh                    # instalar dependencias
  source venv/bin/activate
  python3 bot.py
"""

import os, json, logging
import numpy as np
import pandas as pd
import joblib
import yfinance as yf

from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ConversationHandler, ContextTypes, filters,
)
from config import TELEGRAM_TOKEN

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Rutas — busca bot_data/ en el directorio actual o en el padre ─────────
BASE = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(os.path.join(BASE, "bot_data")):
    DATA = os.path.join(BASE, "bot_data")
elif os.path.isdir(os.path.join(BASE, "..", "bot_data")):
    DATA = os.path.join(BASE, "..", "bot_data")
else:
    raise FileNotFoundError("No se encontró la carpeta bot_data/. "
                            "Debe estar junto a bot.py o en el directorio padre.")

LB_A_KG = 0.453592
FACTOR_CEREZA_VERDE = 5.5

# ═══════════════════════════════════════════════════════════════════════════
# CARGA DE MODELOS Y DATOS
# ═══════════════════════════════════════════════════════════════════════════
def cargar_datos():
    d = {}
    d["arima"] = joblib.load(os.path.join(DATA, "modelos", "arima_model.pkl"))
    logger.info("ARIMA cargado ✓")
    d["rf"] = joblib.load(os.path.join(DATA, "modelos", "rf_model.pkl"))
    logger.info("Random Forest cargado ✓")
    with open(os.path.join(DATA, "modelos", "rf_features.json")) as f:
        d["rf_features"] = json.load(f)
    logger.info(f"Features RF: {d['rf_features']}")
    d["brecha"] = pd.read_csv(os.path.join(DATA, "datos", "brecha.csv"), index_col=0)
    logger.info(f"Brecha: {len(d['brecha'])} años")
    d["rend_estado"] = pd.read_csv(os.path.join(DATA, "datos", "rendimientos_estado.csv"), index_col=0)
    logger.info(f"Rendimientos estado: {len(d['rend_estado'])}")
    d["rend_muni"] = pd.read_csv(os.path.join(DATA, "datos", "rendimientos_municipio.csv"), index_col=[0,1])
    logger.info(f"Rendimientos municipio: {len(d['rend_muni'])}")
    with open(os.path.join(DATA, "datos", "plan_optimo.json")) as f:
        d["plan"] = json.load(f)
    logger.info("Plan óptimo cargado ✓")
    with open(os.path.join(DATA, "datos", "metadata.json")) as f:
        d["meta"] = json.load(f)
    logger.info(f"Metadata: {d['meta']['fecha_ejecucion']}")
    return d

D = cargar_datos()

# ═══════════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ═══════════════════════════════════════════════════════════════════════════
def obtener_precio_actual():
    try:
        kc = yf.download("KC=F", period="5d", progress=False, auto_adjust=False)
        if isinstance(kc.columns, pd.MultiIndex):
            kc.columns = kc.columns.get_level_values(0)
        precio_usd = float(kc["Close"].dropna().iloc[-1]) / 100.0
        fx = yf.download("USDMXN=X", period="5d", progress=False, auto_adjust=False)
        if isinstance(fx.columns, pd.MultiIndex):
            fx.columns = fx.columns.get_level_values(0)
        tc = float(fx["Close"].dropna().iloc[-1])
        precio_mxn = precio_usd * tc / LB_A_KG
        return precio_usd, tc, precio_mxn
    except Exception as e:
        logger.error(f"Error precios: {e}")
        return None, None, None

def obtener_pronosticos():
    r = {}
    for etiq, h in [("3 meses", 3), ("6 meses", 6), ("12 meses", 12)]:
        pt, ci = D["arima"].predict(n_periods=h, return_conf_int=True)
        pt = np.asarray(pt)
        r[etiq] = {"punto": float(pt[-1]), "inf": float(ci[-1,0]), "sup": float(ci[-1,1])}
    return r

def obtener_senal_ml(precio_usd, tc):
    try:
        kc = yf.download("KC=F", period="3mo", progress=False, auto_adjust=False)
        if isinstance(kc.columns, pd.MultiIndex):
            kc.columns = kc.columns.get_level_values(0)
        serie = kc["Close"].dropna() / 100.0
        if len(serie) < 31:
            return None, "Datos insuficientes"
        row = {
            "usdmxn": tc,
            "lag_1": float(serie.iloc[-2]),
            "ret_1": float(serie.pct_change().iloc[-1]),
            "lag_7": float(serie.iloc[-8]) if len(serie) > 8 else float(serie.iloc[0]),
            "ret_7": float(serie.pct_change(7).iloc[-1]),
            "lag_30": float(serie.iloc[-31]) if len(serie) > 31 else float(serie.iloc[0]),
            "ret_30": float(serie.pct_change(30).iloc[-1]),
            "vol_30": float(serie.pct_change().rolling(30).std().iloc[-1]),
            "mes": int(pd.Timestamp.now().month),
            "trimestre": int((pd.Timestamp.now().month - 1) // 3 + 1),
        }
        X = pd.DataFrame([row])[D["rf_features"]]
        proba = float(D["rf"].predict_proba(X)[0, 1])
        return proba, "ALZA" if proba > 0.5 else "BAJA"
    except Exception as e:
        logger.error(f"Error señal ML: {e}")
        return None, "Error"

def obtener_media_252():
    try:
        kc = yf.download("KC=F", period="2y", progress=False, auto_adjust=False)
        if isinstance(kc.columns, pd.MultiIndex):
            kc.columns = kc.columns.get_level_values(0)
        return float((kc["Close"] / 100.0).rolling(252).mean().dropna().iloc[-1])
    except:
        return None

# ═══════════════════════════════════════════════════════════════════════════
# COMANDOS
# ═══════════════════════════════════════════════════════════════════════════
async def cmd_ayuda(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "☕ *Mercado Justo — Bot de Café*\n\n"
        "📊 /precio — Precio actual del café\n"
        "📈 /pronostico — Pronóstico a 3, 6 y 12 meses\n"
        "✅ /recomendacion — VENDER o MANTENER\n"
        "🤖 /senal — Señal de Machine Learning\n"
        "🔄 /brecha — Brecha productor vs internacional\n"
        "🌱 /cultivo — Plan óptimo de cultivo\n"
        "🧮 /calcular — Calculadora personalizada\n"
        "📋 /resumen — Todo en un mensaje\n"
        "❓ /ayuda — Este mensaje\n\n"
        f"_Modelos actualizados: {D['meta']['fecha_ejecucion']}_",
        parse_mode="Markdown")

async def cmd_precio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Consultando precio en vivo...")
    p, tc, pmxn = obtener_precio_actual()
    if p is None:
        await update.message.reply_text("❌ No se pudo obtener el precio.")
        return
    await update.message.reply_text(
        f"☕ *Precio actual del café KC=F*\n\n"
        f"💵 {p:.2f} USD/lb\n"
        f"🇲🇽 {pmxn:.2f} MXN/kg (verde)\n"
        f"💱 Tipo de cambio: {tc:.2f} MXN/USD\n\n"
        f"_Fuente: Yahoo Finance (en vivo)_",
        parse_mode="Markdown")

async def cmd_pronostico(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Calculando pronósticos...")
    p, tc, pmxn = obtener_precio_actual()
    prons = obtener_pronosticos()
    lineas = ["📈 *Pronósticos ARIMA del precio KC=F*\n"]
    for etiq, v in prons.items():
        cambio = ((v["punto"] - p) / p * 100) if p else 0
        flecha = "▲" if cambio > 0 else "▼"
        mxn = v["punto"] * tc / LB_A_KG if tc else 0
        lineas.append(
            f"*{etiq}:* {v['punto']:.2f} USD/lb ({mxn:.1f} MXN/kg)\n"
            f"   {flecha}{abs(cambio):.1f}%  IC:[{v['inf']:.2f}, {v['sup']:.2f}]")
    if p: lineas.append(f"\n_Precio actual: {p:.2f} USD/lb ({pmxn:.1f} MXN/kg)_")
    await update.message.reply_text("\n".join(lineas), parse_mode="Markdown")

async def cmd_recomendacion(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Analizando mercado...")
    p, tc, _ = obtener_precio_actual()
    prons = obtener_pronosticos()
    proba, senal = obtener_senal_ml(p, tc)
    media = obtener_media_252()
    if p is None or media is None:
        await update.message.reply_text("❌ No se pudo obtener datos.")
        return
    arriba = p > media
    baja = prons["3 meses"]["punto"] < p
    if arriba and baja:
        accion, razon = "VENDER ✅", "Precio arriba de la media y pronóstico a la baja."
    elif arriba:
        accion, razon = "MANTENER ⏳", "Tendencia alcista vigente."
    else:
        accion, razon = "MANTENER ⏳", "Precio bajo la media. Mejor esperar."
    await update.message.reply_text(
        f"✅ *Recomendación — Mercado Justo*\n\n"
        f"Precio actual: {p:.2f} USD/lb\nMedia 252 días: {media:.2f} USD/lb\n"
        f"Pronóstico 3m: {prons['3 meses']['punto']:.2f} USD/lb\n"
        f"Señal ML: {senal} ({proba:.0%})\n\n"
        f"🎯 *Acción: {accion}*\n_{razon}_",
        parse_mode="Markdown")

async def cmd_senal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Consultando Random Forest...")
    p, tc, _ = obtener_precio_actual()
    proba, senal = obtener_senal_ml(p, tc)
    if proba is None:
        await update.message.reply_text(f"❌ {senal}")
        return
    emoji = "🟢" if senal == "ALZA" else "🔴"
    barra = "█" * int(proba * 20) + "░" * (20 - int(proba * 20))
    await update.message.reply_text(
        f"🤖 *Señal de Machine Learning*\n\n"
        f"{emoji} Tendencia: *{senal}*\n"
        f"Probabilidad de alza: *{proba:.1%}*\n\n"
        f"`[{barra}]`\n\n"
        f"_Modelo: Random Forest (300 árboles)_",
        parse_mode="Markdown")

async def cmd_brecha(update: Update, context: ContextTypes.DEFAULT_TYPE):
    br = D["brecha"]
    ult = br.index[-1]
    prod = float(br.loc[ult, "productor_mxn_kg"])
    equiv = prod * FACTOR_CEREZA_VERDE
    fut = float(br.loc[ult, "futuros_mxn_kg"])
    gap = float(br.loc[ult, "brecha_ajustada_pct"])
    await update.message.reply_text(
        f"🔄 *Brecha de precios — {int(ult)}*\n\n"
        f"🌿 Productor (cereza): {prod:.2f} MXN/kg\n"
        f"🌿 Equivalente verde (×5.5): {equiv:.2f} MXN/kg\n"
        f"🌍 ICE Futures (verde): {fut:.2f} MXN/kg\n\n"
        f"📉 Brecha ajustada: *{gap:.1f}%*\n"
        f"→ El productor captura ~{100-gap:.0f}% del precio internacional",
        parse_mode="Markdown")

async def cmd_cultivo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    p = D["plan"]
    ha = p["plan_hectareas"]
    params = p.get("parametros", {})
    lineas = ["🌱 *Plan óptimo de cultivo*\n"]
    for tipo, hectareas in ha.items():
        if hectareas > 0.01:
            precio = params.get("precios_mxn_kg", {}).get(tipo, 0)
            rend = params.get("rendimiento_kg_ha", {}).get(tipo, 0)
            lineas.append(f"  ☕ *{tipo.title()}:* {hectareas:.1f} ha → "
                          f"{hectareas*rend:,.0f} kg → MXN {hectareas*rend*precio:,.0f}")
    lineas.append(f"\n💰 *Ingreso total: MXN {p['ingreso_total']:,.0f}*")
    await update.message.reply_text("\n".join(lineas), parse_mode="Markdown")

async def cmd_resumen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Preparando resumen completo...")
    p, tc, pmxn = obtener_precio_actual()
    prons = obtener_pronosticos()
    proba, senal = obtener_senal_ml(p, tc)
    gap = float(D["brecha"]["brecha_ajustada_pct"].iloc[-1])
    media = obtener_media_252()
    arriba = (p > media) if p and media else False
    baja = (prons["3 meses"]["punto"] < p) if p else False
    accion = "VENDER ✅" if arriba and baja else "MANTENER ⏳"
    pl = D["plan"]
    t = (f"☕ *MERCADO JUSTO — Resumen*\n\n"
         f"📊 *Precio:* {p:.2f} USD/lb ({pmxn:.1f} MXN/kg)\n"
         f"📈 *3m:* {prons['3 meses']['punto']:.2f} | *6m:* {prons['6 meses']['punto']:.2f} | "
         f"*12m:* {prons['12 meses']['punto']:.2f} USD/lb\n"
         f"🤖 *ML:* {senal} ({proba:.0%})\n"
         f"🔄 *Brecha:* {gap:.1f}%\n"
         f"🎯 *{accion}*\n\n"
         f"🌱 *Cultivo óptimo:*\n")
    for tipo, ha in pl["plan_hectareas"].items():
        if ha > 0.01: t += f"  {tipo.title()}: {ha:.1f} ha\n"
    t += f"  Ingreso: MXN {pl['ingreso_total']:,.0f}\n"
    t += f"\n_Actualizado: {D['meta']['fecha_ejecucion']}_"
    await update.message.reply_text(t, parse_mode="Markdown")

# ═══════════════════════════════════════════════════════════════════════════
# CALCULADORA (/calcular)
# ═══════════════════════════════════════════════════════════════════════════
ESTADO, MODO, MUNICIPIO, HECTAREAS = range(4)
ESTADOS = ["Chiapas", "Oaxaca", "Veracruz"]

async def calc_inicio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🧮 *Calculadora de producción*\n\n¿En qué estado estás?",
        reply_markup=ReplyKeyboardMarkup([ESTADOS], one_time_keyboard=True, resize_keyboard=True),
        parse_mode="Markdown")
    return ESTADO

async def calc_estado(update: Update, context: ContextTypes.DEFAULT_TYPE):
    estado = update.message.text.strip().title()
    if estado not in ESTADOS:
        await update.message.reply_text("❌ Elige Chiapas, Oaxaca o Veracruz.")
        return ESTADO
    context.user_data["estado"] = estado
    await update.message.reply_text(
        f"✅ *{estado}*\n\n¿Promedio del estado o municipio específico?",
        reply_markup=ReplyKeyboardMarkup([["Promedio del estado", "Buscar municipio"]],
                                         one_time_keyboard=True, resize_keyboard=True),
        parse_mode="Markdown")
    return MODO

async def calc_modo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if "municipio" in update.message.text.lower() or "buscar" in update.message.text.lower():
        await update.message.reply_text("📍 Escribe el nombre de tu municipio:",
                                        reply_markup=ReplyKeyboardRemove())
        return MUNICIPIO
    estado = context.user_data["estado"]
    re = D["rend_estado"]
    if estado in re.index:
        context.user_data["rendimiento"] = float(re.loc[estado, "rendimiento_ton_ha"])
        context.user_data["precio_local"] = float(re.loc[estado, "precio_mxn_kg"])
        context.user_data["ubicacion"] = estado
    else:
        await update.message.reply_text("❌ No hay datos para ese estado.")
        return ConversationHandler.END
    await update.message.reply_text(
        f"📊 Rendimiento en {estado}: *{context.user_data['rendimiento']:.2f} ton/ha*\n\n"
        f"¿Cuántas hectáreas vas a sembrar?",
        reply_markup=ReplyKeyboardRemove(), parse_mode="Markdown")
    return HECTAREAS

async def calc_municipio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    muni_input = update.message.text.strip().title()
    estado = context.user_data["estado"]
    rm = D["rend_muni"]
    try:
        munis = [m for m in rm.loc[estado].index if muni_input.lower() in m.lower()]
    except KeyError:
        munis = []
    if not munis:
        todos = [(e, m) for e in rm.index.get_level_values(0).unique()
                 for m in rm.loc[e].index if muni_input.lower() in m.lower()]
        if todos:
            estado, muni = todos[0][0], todos[0][1]
            context.user_data["estado"] = estado
            munis = [muni]
        else:
            await update.message.reply_text(f"❌ No encontré *{muni_input}*. Intenta otro nombre.",
                                            parse_mode="Markdown")
            return MUNICIPIO
    muni = munis[0]
    datos = rm.loc[(context.user_data["estado"], muni)]
    context.user_data["rendimiento"] = float(datos["rendimiento_ton_ha"])
    context.user_data["precio_local"] = float(datos["precio_mxn_kg"])
    context.user_data["ubicacion"] = f"{muni}, {context.user_data['estado']}"
    await update.message.reply_text(
        f"📍 *{muni}, {context.user_data['estado']}*\n"
        f"Rendimiento: *{context.user_data['rendimiento']:.2f} ton/ha*\n"
        f"Precio local: *{context.user_data['precio_local']:.2f} MXN/kg*\n\n"
        f"¿Cuántas hectáreas vas a sembrar?", parse_mode="Markdown")
    return HECTAREAS

async def calc_hectareas(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        ha = float(update.message.text.strip().replace(",", "."))
        if ha <= 0 or ha > 10000: raise ValueError
    except ValueError:
        await update.message.reply_text("❌ Escribe un número válido (ej: 12).")
        return HECTAREAS

    rend = context.user_data["rendimiento"]
    precio_local = context.user_data["precio_local"]
    ubicacion = context.user_data["ubicacion"]
    prod_kg = ha * rend * 1000
    ingreso_local = prod_kg * precio_local

    p, tc, pmxn = obtener_precio_actual()
    prons = obtener_pronosticos()

    def ingreso_fut(pron):
        if not p or p == 0: return ingreso_local
        return ingreso_local * (pron / p)

    t = (f"🧮 *Estimación para {ubicacion}*\n*{ha:.1f} hectáreas*\n\n"
         f"📦 *Producción:* {ha*rend:,.1f} ton ({prod_kg:,.0f} kg)\n\n"
         f"💰 *Ingreso por escenario:*\n\n"
         f"   📍 *Precio local (SIAP):*\n"
         f"      {precio_local:.2f} MXN/kg → *MXN {ingreso_local:,.0f}*\n\n")
    if p:
        for etiq in ["3 meses", "6 meses"]:
            cambio = ((prons[etiq]["punto"] - p) / p) * 100
            flecha = "▲" if cambio > 0 else "▼"
            t += (f"   📈 *En {etiq}:*\n"
                  f"      {flecha}{abs(cambio):.1f}% → *MXN {ingreso_fut(prons[etiq]['punto']):,.0f}*\n\n")
    gap = float(D["brecha"]["brecha_ajustada_pct"].iloc[-1])
    t += f"📉 Brecha internacional: {gap:.0f}%\n\n"
    if p and prons["3 meses"]["punto"] > p:
        t += "💡 *MANTENER* si puedes almacenar — mejores precios en 3 meses."
    else:
        t += "💡 Considerar *venta parcial* — sin mejora clara a corto plazo."
    await update.message.reply_text(t, parse_mode="Markdown")
    return ConversationHandler.END

async def calc_cancelar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("❌ Cancelado.", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END

async def msg_default(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Usa /ayuda para ver los comandos disponibles.")

# ═══════════════════════════════════════════════════════════════════════════
# ARRANQUE
# ═══════════════════════════════════════════════════════════════════════════
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    conv = ConversationHandler(
        entry_points=[CommandHandler("calcular", calc_inicio)],
        states={
            ESTADO:    [MessageHandler(filters.TEXT & ~filters.COMMAND, calc_estado)],
            MODO:      [MessageHandler(filters.TEXT & ~filters.COMMAND, calc_modo)],
            MUNICIPIO: [MessageHandler(filters.TEXT & ~filters.COMMAND, calc_municipio)],
            HECTAREAS: [MessageHandler(filters.TEXT & ~filters.COMMAND, calc_hectareas)],
        },
        fallbacks=[CommandHandler("cancelar", calc_cancelar)],
    )
    app.add_handler(CommandHandler("start", cmd_ayuda))
    app.add_handler(CommandHandler("ayuda", cmd_ayuda))
    app.add_handler(CommandHandler("precio", cmd_precio))
    app.add_handler(CommandHandler("pronostico", cmd_pronostico))
    app.add_handler(CommandHandler("recomendacion", cmd_recomendacion))
    app.add_handler(CommandHandler("senal", cmd_senal))
    app.add_handler(CommandHandler("brecha", cmd_brecha))
    app.add_handler(CommandHandler("cultivo", cmd_cultivo))
    app.add_handler(CommandHandler("resumen", cmd_resumen))
    app.add_handler(conv)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, msg_default))
    logger.info("Bot iniciado — esperando mensajes...")
    app.run_polling()

if __name__ == "__main__":
    main()
