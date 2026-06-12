@echo off

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║       Multi-Agent RAG Chatbot — Setup Starting          ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

echo [1/5] Checking Python installation...
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found! Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)
python --version
echo       Python found!
echo.

echo [2/5] Creating virtual environment (.venv)...
IF EXIST .venv (
    echo       .venv already exists, skipping creation.
) ELSE (
    python -m venv .venv
    echo       .venv created successfully!
)
echo.

echo [3/5] Activating virtual environment...
call .venv\Scripts\activate.bat
echo       Activated!
echo.

echo [4/5] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo       pip upgraded!
echo.

echo [5/5] Installing dependencies from requirements.txt...
echo       This may take a few minutes, please wait...
pip install -r requirements.txt
echo.

IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Some packages failed to install. Check the output above.
    pause
    exit /b 1
)

echo Creating data directories...
IF NOT EXIST data\chroma_db mkdir data\chroma_db
IF NOT EXIST data mkdir data
echo       data\ directories ready!
echo.

echo ╔══════════════════════════════════════════════════════════╗
echo ║                  Setup Complete!                        ║
echo ╠══════════════════════════════════════════════════════════╣
echo ║  Next Steps:                                            ║
echo ║                                                         ║
echo ║  1. Make sure your .env file has your GROQ_API_KEY     ║
echo ║                                                         ║
echo ║  2. Activate the virtual environment:                  ║
echo ║     .venv\Scripts\activate                             ║
echo ║                                                         ║
echo ║  3. Run the chatbot:                                    ║
echo ║     streamlit run app.py                               ║
echo ╚══════════════════════════════════════════════════════════╝
echo.
pause
