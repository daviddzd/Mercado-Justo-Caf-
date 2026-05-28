# ☕ Mercado Justo — Predicción de Precios del Café

Análisis y pronóstico de precios para apoyar decisiones de comercialización
de pequeños productores de café en Chiapas, Oaxaca y Veracruz.

## Links

- 📖 [Documentación completa (GitBook)](https://davids-5.gitbook.io/mercado-justo-cafe/)
- 🤖 [Bot de Telegram](https://t.me/mercadojusto_cafe_bot)

## Stack tecnológico

PySpark · ARIMA · Random Forest · PuLP · Plotly · Telegram Bot · Yahoo Finance API

## Estructura

- `notebook/` — Notebook principal (Colab/Jupyter)
- `bot/` — Bot de Telegram (Python)
- `bot_data/` — Modelos y datos exportados
- `documentos/` — Contenido del proyecto
- `capturas/` — Gráficas del análisis

## Cómo ejecutar

### Notebook
1. Abrir `notebook/Mercado_Justo_Cafe_Final.ipynb` en Google Colab o Jupyter
2. Subir `cafe_cereza_2003_2024.xlsx` al entorno
3. Ejecutar todas las celdas

### Bot de Telegram
1. `cd bot/`
2. `cp config.py.example config.py` y agregar tu token
3. `bash setup.sh`
4. `source venv/bin/activate && python3 bot.py`

## quipo

David Pitol Arana
Daniela Hernandez Carrada
Rene De la Cruz Escobar
Daniel Aldair Montiel Macias

Universidad Nacional Rosario Castellanos
Licenciatura en Ciencias de Datos para Negocios — 5° Semestre — 2026-1