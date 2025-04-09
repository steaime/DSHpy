@echo on
call %CONDA_ROOT%\Scripts\activate.bat %CONDA_ROOT%\envs\DSH
cd %DSHpy_ROOT%
pip uninstall dsh
python -m pip install .
echo ...done!

pause