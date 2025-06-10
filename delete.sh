clear
find . -type d -name '__pycache__' -exec rm -r {} + -o -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
python main.py
find . -type d -name '__pycache__' -exec rm -r {} + -o -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
