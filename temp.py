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

LABEL_STUDIO_URL = 'http://localhost:8080'
API_KEY = '49b3cde16cc34ec2f3e4652904a1775844462121'

# Connect to the Label Studio API and check the connection
ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
ls.check_connection()
# for i in allfiles:
for filename in sorted(os.listdir("./Temp")):
    project = ls.start_project(
        title = 'Final Project',
        label_config='''
        <View>
            <Image name="image" value="$ocr"/>

            <Rectangle name="bbox" toName="image" strokeWidth="3"/>
            <Polygon name="poly" toName="image" strokeWidth="3"/>

            <RectangleLabels name="rect-label" toName="image">
              <Label value="Header Block" background="red"/>
              <Label value="Date Block" background="orange"/>
              <Label value="Subject Block" background="yellow"/>
              <Label value="Copy-Forwarded To Block" background="purple"/>
              <Label value="Body Block" background="cyan"/>
              <Label value="Table" background="pink"/>
              <Label value="Stamps/Seals" background="brown"/>
              <Label value="Logos" background="magenta"/>
              <Label value="Sender Information Block" background="teal"/>
              <Label value="Signature" background="lime"/>
            </RectangleLabels>

            <Labels name="aspect" toName="q2">
              <Label value="question" background="#FFA39E"/>
            </Labels>
            <Header value="Please answer these questions:"/>

            <!-- Q1 -->
            <View style="display: grid; grid-template-columns: 1fr 10fr 1fr 3fr; column-gap: 1em">
              <Header value="Q2:"/>
              <Text name="q2" value="What is the Subject of the given Circular?"/>
              <Header value="A2:"/>
              <TextArea name="answer2" toName="q2" rows="1" maxSubmissions="1"/>
            </View>

            <!-- Q2 -->
            <View style="display: grid; grid-template-columns: 1fr 10fr 1fr 3fr; column-gap: 1em">
              <Header value="Q4:"/>
              <Text name="q4" value="Which Organization issued this circular?"/>
              <Header value="A4:"/>
              <TextArea name="answer4" toName="q4" rows="1" maxSubmissions="1"/>
            </View>

            <!-- Q3 -->
            <View style="display: grid; grid-template-columns: 1fr 10fr 1fr 3fr; column-gap: 1em">
              <Header value="Q3:"/>
              <Text name="q3" value="What is the Content of the given Circular?"/>
              <Header value="A3:"/>
              <TextArea name="answer3" toName="q3" rows="1" maxSubmissions="1"/>
            </View>

            <!-- Q6 -->
            <View style="display: grid; grid-template-columns: 1fr 10fr 1fr 3fr; column-gap: 1em">
              <Header value="Q6:"/>
              <Text name="q6" value="What is the Date of Issuance of the Circular?"/>
              <Header value="A6:"/>
              <TextArea name="answer6" toName="q6" rows="1" maxSubmissions="1"/>
            </View>

            <!-- Q7 -->
            <View style="display: grid; grid-template-columns: 1fr 10fr 1fr 3fr; column-gap: 1em">
              <Header value="Q7:"/>
              <Text name="q7" value="What is the Content of the given Circular?"/>
              <Header value="A7:"/>
              <TextArea name="answer7" toName="q7" rows="1" maxSubmissions="1"/>
            </View>

            <!-- Q8 -->
            <View style="display: grid; grid-template-columns: 1fr 10fr 1fr 3fr; column-gap: 1em">
              <Header value="Q8:"/>
              <Text name="q8" value="Who has signed the Given Circular?"/>
              <Header value="A8:"/>
              <TextArea name="answer8" toName="q8" rows="1" maxSubmissions="1"/>
            </View>

            <!-- Q9 -->
            <View style="display: grid; grid-template-columns: 1fr 10fr 1fr 3fr; column-gap: 1em">
              <Header value="Q9:"/>
              <Text name="q9" value="What are the credentials of the Signatory?"/>
              <Header value="A9:"/>
              <TextArea name="answer9" toName="q9" rows="1" maxSubmissions="1"/>
            </View>

            <!-- Q10 -->
            <View style="display: grid; grid-template-columns: 1fr 10fr 1fr 3fr; column-gap: 1em">
              <Header value="Q10:"/>
              <Text name="q10" value="List the People this Circular has been forwarded to along with their designations and organizations."/>
              <Header value="A10:"/>
              <TextArea name="answer10" toName="q10" rows="1" maxSubmissions="1"/>
            </View>

            <!-- Q11 -->
            <View style="display: grid; grid-template-columns: 1fr 10fr 1fr 3fr; column-gap: 1em">
              <Header value="Q11:"/>
              <Text name="q11" value="What is the Serial no. of the Given Circular?"/>
              <Header value="A11:"/>
              <TextArea name="answer11" toName="q11" rows="1" maxSubmissions="1"/>
            </View>

            <TextArea name="transcription" toName="image" editable="true" perRegion="true" required="true" maxSubmissions="1" rows="5"/>
        </View>
        '''
    )

    with open('./experiment_others/Temp.json', mode='r') as f:
        task = f.read()

    jtask = json.loads(task)
    #print(jtask)

    project.import_tasks(tasks=jtask)
    break