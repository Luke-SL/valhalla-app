#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
VENV_PYTHON="$DIR/venv/bin/python"
VENV_STREAMLIT="$DIR/venv/bin/streamlit"

echo "=== RAG Sistema de Consultas a Normas (Llama 3.2) ==="

if [ ! -f "$VENV_PYTHON" ]; then
    echo "--> Ambiente virtual não encontrado. Criando em $DIR/venv..."
    python3 -m venv "$DIR/venv"
    echo "--> Instalando dependências de requirements.txt..."
    "$DIR/venv/bin/pip" install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
    "$DIR/venv/bin/pip" install --no-cache-dir -r "$DIR/requirements.txt"
fi

case "$1" in
    index)
        echo "--> Indexando PDFs de $DIR/documentos..."
        $VENV_PYTHON "$DIR/rag_cli.py" --index --force
        ;;
    ask)
        shift
        $VENV_PYTHON "$DIR/rag_cli.py" --ask "$*"
        ;;
    cli)
        $VENV_PYTHON "$DIR/rag_cli.py"
        ;;
    ui|web|streamlit)
        echo "--> Iniciando interface gráfica Streamlit..."
        $VENV_STREAMLIT run "$DIR/app.py"
        ;;
    *)
        echo "Uso do comando:"
        echo "  ./run.sh index           # Indexa/reindexa os documentos PDF do diretório documentos/"
        echo "  ./run.sh cli             # Abre a interface interativa de chat no terminal"
        echo "  ./run.sh ask \"pergunta\"  # Faz uma pergunta direta pelo terminal"
        echo "  ./run.sh ui              # Abre a interface Web no navegador (Streamlit)"
        ;;
esac
