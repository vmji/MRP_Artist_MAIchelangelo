@echo off 
setlocal enabledelayedexpansion

::Pruefen der Adminrechte
net session >nul 2>&1
if %errorLevel% == 0 (
    goto :ask
)

::Abfrage Skript als Admin auszufuehren
echo Requesting administrative privileges...
echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\getadmin.vbs"
echo UAC.ShellExecute "%~s0", "", "", "runas", 1 >> "%temp%\getadmin.vbs"
"%temp%\getadmin.vbs"
del "%temp%\getadmin.vbs"
exit /b

::Frage ob Installation ausgefuehrt werden soll
:ask
echo "Do you want to install the required files to use MAIchelangelo? (Y/n)"
set /p INSTALLATION=""
::Falls nein, Abbruch des Skripts
if /i "%INSTALLATION%"=="n" goto end
if /i not "%INSTALLATION%"=="Y" goto end

:: Generierung des ssh keys falls noch nicht vorhanden

::Ausfuehren der Installationsskripts, falls diese noch nicht ausgefuehrt wurden call wird verwendet, um Programme auszufuehren und dann weiterzufuehren
:install
set "SCRIPT_DIR=%~dp0"
cd %SCRIPT_DIR%
call installation_script_desktop_link.cmd

:end