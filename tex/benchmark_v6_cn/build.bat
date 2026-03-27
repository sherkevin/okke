@echo off
cd /d "%~dp0"
latexmk -xelatex -interaction=nonstopmode -file-line-error main.tex
if errorlevel 1 exit /b 1
echo OK: main.pdf
