# Instructions for Using the Annotation Pipeline

- This Consists of Three Scripts
    1. setup.py - Sets up the JSONs required for the next steps. Such as configuring the links for the data and the rest
    2. flask_s.py: Creates a Flask server that enables us to interact with the files in the given folder, this must be run, to access the files via label studio
    3. label-studio.py: Configures individual projects by using the JSON's obtained from setup.py with the OCR output. 

- Further Testing Is required