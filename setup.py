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

def create_image_url(filepath, fname):
    filename = os.path.basename(filepath)
    return f'http://127.0.0.1:8081/Temp/{filename}'

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

def process_images_and_save(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all jpg files in the input folder
    jpg_files = Path(input_folder).glob("*.jpg")

    for jpg_file in jpg_files:
        with Image.open(jpg_file) as image:
            model = ocr_predictor(pretrained=True)
            doc = DocumentFile.from_images(jpg_file)
            res = model(doc)
            model_output = res.export()
            task = convert_to_ls(image, model_output, jpg_file.stem)

            # Write to a json in the output folder
            output_path = os.path.join(output_folder, f"{jpg_file.stem}.json")
            with open(output_path, 'w') as outfile:
                json.dump(task, outfile)

# Example usage:
input_folder_path = './Temp'
output_folder_path = './JSONs'
process_images_and_save(input_folder_path, output_folder_path)
 
