@echo off
echo ==========================================
echo    EXXATA MONTE CARLO - REACT (VITE)
echo ==========================================
echo.

:: Verifica se o Node.js está instalado
where node >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERRO] O Node.js nao foi encontrado. Por favor, instale o Node.js para continuar.
    pause
    exit /b
)

if not exist "node_modules" (
    echo [1/2] Instalando dependencias (isso pode levar um momento)...
    call npm install
)

echo.
echo [2/2] Iniciando a plataforma em modo desenvolvimento...
echo A plataforma abrira automaticamente em: http://localhost:5173
echo.

:: Abre o navegador e inicia o vite
start http://localhost:5173
call npm run dev

pause
