@echo off
setlocal enabledelayedexpansion
chcp 65001

:: Load variables from .env file
if exist ".env" (
    for /f "usebackq delims=" %%L in (".env") do (
        set "line=%%L"
        rem Skip comments and empty lines
        if not "!line:~0,1!"=="#" if defined line (
            for /f "tokens=1,* delims==" %%A in ("!line!") do (
                set "%%A=%%B"
            )
        )
    )
)

:: Nutzen username aus .env oder Abfrage wenn dieser nicht festgelegt ist
if not defined USERNAME (
    set /p USERNAME= "Enter your username: " 
) 
echo 'Welcome %USERNAME%!'
set REMOTE_DIR=/mount/point/%USERNAME%/input_images

echo Uploaded images will be stored in '%REMOTE_DIR%'

call ssh %USERNAME%@%MRP_IP% "mkdir -p %REMOTE_DIR%; cd %REMOTE_DIR%/; rm -rf *.png; rm -rf *.img; exit"

:ask
set /p UPLOAD_FILES="Do you want to upload image files for editing from your local machine? (Y/n): "
if /i "%UPLOAD_FILES%"=="n" goto end
if /i not "%UPLOAD_FILES%"=="Y" goto end

:loop
set /p FILE_PATH_FROM="Enter the path to the image file or drag and drop the file (type 'exit' to finish): "
if /i "%FILE_PATH_FROM%"=="exit" goto end
:: Entfernen von Anfuehrungszeichen falls diese vorhanden sind
set "FILE_PATH_FROM=!FILE_PATH_FROM:"=!"
:: Handhaben von Drag and Drop
for /f "tokens=*" %%a in ("!FILE_PATH_FROM!") do set "FILE_PATH_FROM=%%a"

:: Pruefen, ob die Datei existiert
if not exist "!FILE_PATH_FROM!" (
    echo Error: File "!FILE_PATH_FROM!" does not exist. Please try again.
    goto loop
)

: Pruefen ob die Datei eine Bilddatei ist
:: Get file extension
for %%f in ("!FILE_PATH_FROM!") do set "FILE_EXT=%%~xf"

:: Convert extension to lowercase for comparison
set "FILE_EXT=!FILE_EXT!"
if defined FILE_EXT (
    for %%a in (A B C D E F G H I J K L M N O P Q R S T U V W X Y Z) do (
        set "FILE_EXT=!FILE_EXT:%%a=%%a!"
    )
)

:: Check if file extension is allowed
set VALID_EXT=0
if /i "!FILE_EXT!"==".png" set VALID_EXT=1
if /i "!FILE_EXT!"==".jpg" set VALID_EXT=1
if /i "!FILE_EXT!"==".jpeg" set VALID_EXT=1
if /i "!FILE_EXT!"==".img" set VALID_EXT=1
if /i "!FILE_EXT!"==".bmp" set VALID_EXT=1
if /i "!FILE_EXT!"==".gif" set VALID_EXT=1
if /i "!FILE_EXT!"==".tiff" set VALID_EXT=1

if !VALID_EXT!==0 (
    echo Error: Only image files are allowed ^(.png, .jpg, .jpeg, .img, .bmp, .gif, .tiff^)
    echo Your file has extension: !FILE_EXT!
    goto loop
)

echo "Copying data from: !FILE_PATH_FROM! to:%REMOTE_DIR%/"
scp "!FILE_PATH_FROM!" %USERNAME%@%MRP_IP%:%REMOTE_DIR%/
goto loop
:end

:: erstellen eines temporaeren dictionaries von welchem aus nur die heruntergeladenen Bilder geoefffnet werden
mkdir "C:\Users\Public\Pictures\MAIchelangelo" 2>nul
set OUTPUT_DIR=/mount/point/%USERNAME%/generated_pictures/
echo Generated images will be downloaded from %OUTPUT_DIR%

ssh %USERNAME%@%MRP_IP% "mkdir -p %OUTPUT_DIR%; cd %OUTPUT_DIR%; rm -rf *.png; /mount/point/veith/.venv/bin/python /mount/point/veith/MRP_Artist_MAIchelangelo/Picture_Generator.py  %REMOTE_DIR% %OUTPUT_DIR%"
scp -r "%USERNAME%@%MRP_IP%:%OUTPUT_DIR%*.png" "C:\Users\Public\Pictures\MAIchelangelo"
"Image(s) saved to C:\Users\Public\Pictures"
setlocal enabledelayedexpansion
for %%f in (C:\Users\Public\Pictures\MAIchelangelo\*.png) do ("%%f")
endlocal
::verschieben der heruntergeladenen Bilder
move C:\Users\Public\Pictures\MAIchelangelo\* C:\Users\Public\Pictures
rmdir C:\Users\Public\Pictures\MAIchelangelo
echo It is recommended to move the generated images to your preferred directory

pause