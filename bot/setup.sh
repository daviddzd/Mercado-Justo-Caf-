#!/bin/bash
# ── Instalación del bot Mercado Justo en Ubuntu Server ────────────────────
# Ejecutar: bash setup.sh

echo "☕ Configurando Mercado Justo Bot..."

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 no encontrado. Instalando..."
    sudo apt update && sudo apt install -y python3 python3-pip python3-venv
fi

echo "📦 Creando entorno virtual..."
python3 -m venv venv
source venv/bin/activate

echo "📦 Instalando dependencias..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Instalación completa."
echo ""
echo "Próximos pasos:"
echo "  1) Asegúrate de que la carpeta bot_data/ esté al mismo nivel que bot.py"
echo "  2) Copia config.py.example como config.py y pon tu token"
echo "     cp config.py.example config.py"
echo "  3) Ejecuta: source venv/bin/activate && python3 bot.py"
echo ""
echo "Para ejecutar en segundo plano:"
echo "  nohup python3 bot.py > bot.log 2>&1 &"
