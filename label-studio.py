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

json_path = './JSONs'

#########################  LABEL STUDIO  #########################
LABEL_STUDIO_URL = 'http://localhost:8080'
API_KEY = '49b3cde16cc34ec2f3e4652904a1775844462121'

# Connect to the Label Studio API and check the connection
ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
ls.check_connection()
# for i in allfiles:
for filename in sorted(os.listdir("./Temp")):
    if filename.endswith(".jpg"):
        print("File Name", filename)
        project = ls.start_project(
            title = filename,
            label_config='''
            <View>
            <Image name="image" value="$ocr"/>

            <Labels name="label" toName="image">
                <Label value="Text" background="green"/>
                <Label value="field" background="blue"/>
                <Label value="value" background="yellow"/>
            </Labels>

            <Rectangle name="bbox" toName="image" strokeWidth="3"/>
            <Polygon name="poly" toName="image" strokeWidth="3"/>

            <TextArea name="transcription" toName="image" editable="true" perRegion="true" required="true" maxSubmissions="1" rows="5" placeholder="Recognized Text" displayMode="region-list"/>
            </View>
            '''
        )
        without_extension = filename.split('.')[0]

        with open(str(json_path) + '/' + str(without_extension) + '.json', mode='r') as f:
            task = f.read()

        jtask = json.loads(task)
    #print(jtask)

        project.import_tasks(tasks=jtask)