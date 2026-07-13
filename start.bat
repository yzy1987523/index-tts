@echo off
chcp 65001 >nul
REM ============================================================
REM IndexTTS-2.0 MVP 一键启动
REM ============================================================
echo [启动] IndexTTS-2.0 MVP ...
echo [工作区] %~dp0
cd /d "%~dp0"

REM 检查模型
if not exist "checkpoints\gpt.pth" (
    echo [警告] 模型未下载,正在从 ModelScope 下载...
    uv run modelscope download --model IndexTeam/IndexTTS-2 --local_dir ./checkpoints
    if errorlevel 1 (
        echo [错误] 模型下载失败
        pause
        exit /b 1
    )
)

REM 启动 UI
echo [启动] Gradio UI 端口 9880 ...
uv run python app.py --port 9880

pause