# Instructions for Using the Annotation Pipeline

These are the instructions for using the Annotation Pipeline, with DocTR, for label-studio

- This Consists of Three Scripts
    1. DocTr.py:
        - This file basically generates a JSON for the given dataset (with the OCR'ed outputs and bboxes) which is of the format of LabelStudio.
        - It has the options for (First/Middle/Last) Pages, built in to it. 
    2. Flask_S.py:
        - This initiates a Flask server on the port of our choice, which in this case is 8081. This is required since, LabelStudio only accepts stuff in this format. 
        - The Home Directory is the one mentioned in the flask_s.py, and all paths are constructed relative to it. (Can be accessed via localhost)
    3. Label-Studio.py: 
        - This is used to Set up the Annotation Pipeline, on label-studio. This is a one-time thing.
        - Here as well you need to adjust the (First/Middle/Last) pages.

- Always Have Three

- How to start Annotating: 
    0. Collect your Data, and store it in 3 folders
        - First
        - Middle
        - Last
    1. Run the Label-Studio.py Script, for all 3 folders
    2. Run the Flask Script, to set up a Python Server. 
    3. Run the Command, label-studio on your local system. 

- Points To Be Noted:
    1. The Data Will Not Load Properly unless and until 