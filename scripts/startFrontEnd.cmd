@echo off
setlocal

set "WORKINGDIRECTORY=C:\Users\%USERNAME%\Documents\Dev\GitHub\RAGPDFTeachingDemo"

set "VENV=C:\WPy64-3.13.12\notebooks\Envs\haicon26"

cd /d "%WORKINGDIRECTORY%"
call "%VENV%\Scripts\activate.bat"

rem optional sanity checks
python -V
where python

rem To start Jupyter or any other service in this venv
::jupyter notebook

streamlit run frontend/app.py