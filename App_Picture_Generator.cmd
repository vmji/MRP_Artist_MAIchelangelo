@echo off
chcp 65001
set /p USERNAME= "Enter your username: "
echo '%USERNAME%'
:: erstellen eines temporären dictionaries von welchem aus nur die heruntergeladenen Bilder geöfffnet werden
mkdir "C:\Users\Public\Pictures\MAIchelangelo" 2>nul
set REMOTE_DIR=/mount/point/%USERNAME%/generated_pictures/
echo Generated images will be downloaded from %REMOTE_DIR%

ssh %USERNAME%@10.246.58.13 "mkdir -p %REMOTE_DIR%; cd %REMOTE_DIR%; rm -rf *.png; /mount/point/veith/.venv/bin/python /mount/point/veith/MRP_Artist_MAIchelangelo/Picture_Generator.py %REMOTE_DIR%"
scp -r %USERNAME%@10.246.58.13:%REMOTE_DIR%*.png C:\Users\Public\Pictures\MAIchelangelo
"Image(s) saved to C:\Users\Public\Pictures"
setlocal enabledelayedexpansion
for %%f in (C:\Users\Public\Pictures\MAIchelangelo\*.png) do ("%%f")
endlocal
::verschieben der heruntergeladenen Bilder
move C:\Users\Public\Pictures\MAIchelangelo\* C:\Users\Public\Pictures
rmdir C:\Users\Public\Pictures\MAIchelangelo
echo It is recommended to move the generated images to your preferred directory.
pause