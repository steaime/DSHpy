@echo on
call C:\Users\steaime\miniconda3\Scripts\activate.bat C:\Users\steaime\miniconda3\envs\DSH
cd C:\Users\steaime\Documents\Codes\DSH\DSHpy
pip uninstall dsh
python -m pip install .
echo ...done!

pause