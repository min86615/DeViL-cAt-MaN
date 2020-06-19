# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 08:21:13 2020

@author: Username
"""
import os
import argparse
# import datetime
# import json
# import random
# import time
# from pathlib import Path

# import numpy as np
import torch
# from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataset import Dataset
# import datasets
# import util.misc as utils
# from datasets import build_dataset, get_coco_api_from_dataset
# from engine import evaluate, train_one_epoch
from models import build_model
import torchvision.transforms as T
from PIL import Image, ImageDraw
import torchvision
# import torchvision.transforms.functional as TF
from torchvision.ops.boxes import batched_nms

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='transfer_dataset')
    parser.add_argument('--coco_path', default='data', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='outputs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=1, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def detect(im, model, device):
    # mean-std normalize the input image (batch-size: 1)
    transform = T.Compose([
    # T.Resize(500),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    img = transform(im).unsqueeze(0).to(device)

    model.to(device)
    # propagate through the model
    outputs = model(img)
    # keep only predictions with confidence above threshold
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1].cpu()
    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, ].cpu(), im.size)
    return probas, bboxes_scaled

def filter_boxes(scores, boxes, confidence=0.7, apply_nms=True, iou=0.5):
    keep = scores.max(-1).values > confidence
    scores, boxes = scores[keep], boxes[keep]

    if apply_nms:
        top_scores, labels = scores.max(-1)
        keep = batched_nms(boxes, top_scores, labels, iou)
        scores, boxes = scores[keep], boxes[keep]

    return scores, boxes

class TestDataset(Dataset):
    def __init__(self, test_folder, transform=None):
        ##############################################
        ### Initialize paths, transforms, and so on
        ##############################################
        self.transform = transform
        self.test_image_list = os.listdir(test_folder)

        # load image path and annotations
        self.imgs = [os.path.join(test_folder, i) for i in self.test_image_list]
        # print(self.imgs)

        
    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        ##############################################
        imgpath = self.imgs[index]
        # img = Image.open(imgpath).convert('RGB')
        img = Image.open(imgpath)
        if self.transform is not None:
            img = self.transform(img)
            
        # img = img.ToTensor()
        img = torchvision.transforms.ToTensor()(img)
        
        return img

    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return len(self.test_image_list)

def test_main(args, model_path, test_folder):
    device = torch.device(args.device)
    model, criterion, postprocessors = build_model(args)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval().to(device)
    
    
    # test_dataset = TestDataset(test_folder)
    # data_loader_test = DataLoader(test_dataset, 1, drop_last=False, num_workers=0)
    
    test_image_list = os.listdir(test_folder)
    
    for image_name in test_image_list:
        image = Image.open(os.path.join(test_folder, image_name)).convert("RGB")
        print(image_name)
        # print(image.size)


        scores, boxes = detect(image, model, device)
        scores, boxes = filter_boxes(scores, boxes)
        
        scores = scores.data.numpy()
        boxes = boxes.data.numpy()
        
        class_id = scores.argmax()
        
        print("Predict class : ", class_id)
        # label = CLASSES[class_id]
        
        
        print(scores, boxes)
        print("*"*10)
    
        # transform = T.Compose([T.Resize(500)])
        # image = transform(image)
        draw = ImageDraw.Draw(image)
        for i in boxes:
            x0, y0, x1, y1 = i
            # draw.rectangle([(x0, y0), (x1, y1)], fill=True)
            # draw.line([(x0, y0), (x0, y1), (x1, y0), (x1, y1), (x0, y1), (x1, y1), (x0, y0), (x1, y0)], fill = (255,0,0), width = 5)
            draw.line([(x0, y0), (x0, y1)], fill = (255,0,0), width = 5)
            draw.line([(x1, y0), (x1, y1)], fill = (255,0,0), width = 5)
            draw.line([(x0, y1), (x1, y1)], fill = (255,0,0), width = 5)
            draw.line([(x0, y0), (x1, y0)], fill = (255,0,0), width = 5)
        image.show()
    
    
    return scores, boxes
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR testing script', parents=[get_args_parser()])
    args = parser.parse_args()
    model_path = "./outputs/checkpoint0099.pth"
    test_folder = "data\\raccoon_val"
    test_folder = "data\\check"
    scores, boxes = test_main(args, model_path, test_folder)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    