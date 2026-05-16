@echo off
echo Creazione ambiente virtuale...
python -m venv .venv
call .venv\Scripts\activate

echo Aggiornamento pip e installazione dipendenze...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo Creazione struttura cartelle...
mkdir data\raw 2>NUL
mkdir data\processed 2>NUL
mkdir notebooks 2>NUL
mkdir models 2>NUL
mkdir configs 2>NUL
mkdir logs 2>NUL
mkdir src 2>NUL
mkdir tests 2>NUL

echo Verifica installazione...
python -c "import pymc, lightgbm, optuna, pandas; print('✅ Stack verificato con successo!')"