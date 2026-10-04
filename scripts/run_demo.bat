@echo off
set COUNT=%1
if "%COUNT%"=="" set COUNT=500
set SEED=%2
if "%SEED%"=="" set SEED=20261004
set OUT=%3
if "%OUT%"=="" set OUT=artifacts\demo
python run_demo.py --count %COUNT% --seed %SEED% --out "%OUT%"
if errorlevel 1 exit /b %errorlevel%
echo Demo dataset written to %OUT%

