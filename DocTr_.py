from pathlib import Path
from uuid import uuid4
from label_studio_sdk import Client
from label_studio_sdk import Project
from label_studio_sdk.data_manager import Column, Filters, Operator, Type
from datetime import datetime
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import re
import sys
import time
import os 
from os.path import join
from xml.dom import minidom
import json
# from pdf2image import convert_from_path
from PIL import Image
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import LayerNorm as BertLayerNorm
import torch
import torchvision
import json
import os
from PIL import Image, ImageDraw, ImageFont
import cv2

# Perform OCR on image

import matplotlib.pyplot as plt
import os
import json
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import numpy as np

json_path = '../experiment_others/'
if not os.path.exists(json_path):
    os.makedirs(json_path)

def create_image_url(filepath,fname):
    """
    Label Studio requires image URLs, so this defines the mapping from filesystem to URLs
    if you use ./serve_local_files.sh <my-images-dir>, the image URLs are localhost:8081/filename.png
    Otherwise you can build links like /data/upload/filename.png to refer to the files
    """
    filename = os.path.basename(filepath)
    return f'http://127.0.0.1:8081/{fname}/{filename}'

def convert_to_ls(image, model_output, fname):
    img_width, img_height = image.size
    results = []
    all_scores = []
    for block in model_output['pages'][0]['blocks']:
        for line in block['lines']:      
          for word in line['words']:
              all_words = []
              confidences = [] 
              all_words.append(word['value'])
              confidences.append(word['confidence'])
              width = word['geometry'][1][0] - word['geometry'][0][0]
              height = word['geometry'][1][1] - word['geometry'][0][1]
              bbox = {
                'x': 100 * word['geometry'][0][0],
                'y': 100 * word['geometry'][0][1],
                'width': 100 * width,
                'height': 100 * height,
                'rotation': 0
            }
              region_id = str(uuid4())[:10]    
              bbox_result = {
                'id': region_id,'from_name': 'bbox', 'to_name': 'image', 
                'type': 'rectangle', 'value': bbox
              }  
              transcription_result = {
                  'id': region_id, 'from_name': 'transcription', 'to_name': 'image',
                'type': 'textarea','value': dict(text=all_words, **bbox), 'score': sum(confidences)/len(confidences)
              }
              results.extend([bbox_result, transcription_result])
              all_scores.append(sum(confidences)/len(confidences))

    return {
      'data': {
        'ocr': create_image_url(image.filename, fname)
      },
      'predictions': [{
        'result': results,
        'score': sum(all_scores) / len(all_scores) if all_scores else 0
      }]
    }

tasks=[]

type = 'First'

for f in sorted(Path(f'../{type}').glob('*.png')):
  print("Entered Loop!!!")
  if f.name.endswith(".png"):
      with Image.open(f.absolute()) as image:
        # use DocTr
        model = ocr_predictor(pretrained=True)
        doc = DocumentFile.from_images(f)
        res = model(doc)
        model_output = res.export()

        task = convert_to_ls(image,model_output,type)
        tasks.append(task)

# create a file to import into Label Studio
with open(str(json_path) + type +'.json', mode='w') as f:
    json.dump(tasks, f, indent=2)

#####################################################################################################
# LABEL_STUDIO_URL = 'http://localhost:8080'
# API_KEY = '49b3cde16cc34ec2f3e4652904a1775844462121'

# Connect to the Label Studio API and check the connection
# ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
# ls.check_connection()
# # for i in allfiles:
# for filename in sorted(os.listdir("./Temp")):
#     project = ls.start_project(
#         title = filename,
#         label_config='''
#         <View>
#         <Image name="image" value="$ocr"/>

#         <Labels name="label" toName="image">
#             <Label value="Text" background="green"/>
#             <Label value="field" background="blue"/>
#             <Label value="value" background="yellow"/>
#         </Labels>

#         <Rectangle name="bbox" toName="image" strokeWidth="3"/>
#         <Polygon name="poly" toName="image" strokeWidth="3"/>

#         <TextArea name="transcription" toName="image" editable="true" perRegion="true" required="true" maxSubmissions="1" rows="5" placeholder="Recognized Text" displayMode="region-list"/>
#         </View>
#         '''
#     )

#     with open('./experiment_others/Temp.json', mode='r') as f:
#         task = f.read()

#     jtask = json.loads(task)
#     #print(jtask)

#     project.import_tasks(tasks=jtask)
#     break