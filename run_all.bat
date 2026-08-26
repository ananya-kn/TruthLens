@echo off
REM Run all scripts in order

python create_env.py

echo Running stage_1.py ...
python stage_1.py

python merge_sessions.py

python create_folders.py

echo Running stage_2.py ...
python stage_2.py

pause
