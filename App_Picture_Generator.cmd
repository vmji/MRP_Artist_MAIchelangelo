@echo off
set /p USERNAME= "Enter your username: "
echo '%USERNAME%'
set REMOTE_DIR=/mount/point/%USERNAME%/generated_pictures/
echo Generated images will be downloaded from %REMOTE_DIR%

ssh %USERNAME%@10.246.58.13 "mkdir -p %REMOTE_DIR%; cd %REMOTE_DIR%; rm -rf *.png; /home/veith/.conda/envs/venv_imagen/bin/python /mount/point/veith/App_Picture_Generator/Picture_Generator.py %REMOTE_DIR%"
scp -r %USERNAME%@10.246.58.13:%REMOTE_DIR%*.png C:\Users\Public\Pictures
"Image(s) saved to C:\Users\Public\Pictures"
setlocal enabledelayedexpansion
for %%f in (C:\Users\Public\Pictures\*.png) do ("%%f")
endlocal
echo It is recommended to move the generated images to your preferred directory.
pause