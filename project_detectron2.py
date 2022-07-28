'''
authority : sanha Hwang
Project : To make 2 mask of clothes

'''
#%%
import torch, torchvision
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os, json, random

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.projects import deeplab
from detectron2.projects import point_rend
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
# %%

path_ = "./stylebot/docker-workspace/data"
#os.chdir("./stylebot_mask/stylebot/docker-workspace/data")
print(os.getcwd())

#coco_metadata = MetadataCatalog.get("coco_2017_val")

# %%

data_path = os.listdir(path_)

a = pd.DataFrame(data_path)
drop_list = a[0].str.contains("json|yml|yaml|file|.DS_Store")

data_path_t = a[~drop_list][0].to_list()
data_path_t

#%% Original clothes Image Path

image_path = [path_ + '/' + i + "/original/" for i in data_path_t]
image_path

# %% sample data
temp = os.listdir(image_path[0])

# %%
temp_image = cv2.imread(image_path[0] + '/' + temp[0])
plt.figure(figsize=(20,20))
im_ = cv2.cvtColor(temp_image, cv2.COLOR_RGB2BGR)
plt.imshow(im_)

im_.shape
#%%
jsonpath = "/home/user/project/stylebot_mask/stylebot/docker-workspace/data/" 
json_file = os.path.join(jsonpath, "stylebot.json")

with open(json_file) as f:
  imgs_anns = json.load(f)

imgs_anns 

"""
- annotation은 list, list 안에 또 딕셔너리 <<imgs_anns["annotations"][0]>> segmentation, iscrowd, image_id, category_id, id, bbox, area
- imges는 리스트 -> 딕셔너리 ->file_name, height, width,id

"""
# %% Make Custom Dataset

def get_dataset_dict(img_dir,d = None):
    try:
        json_file = os.path.join(img_dir, "style_"+d+".json")
        with open(json_file) as f:
            imgs_anns = json.load(f)
        
        return imgs_anns
    except:
        json_file = os.path.join(img_dir)
        with open(json_file) as f:
            imgs_anns = json.load(f)
        return imgs_anns
ataset_dicts = get_dataset_dict(jsonpath, d ="train")

#%%
from detectron2.structures import BoxMode
def get_dataset_dicts(img_dir,d = None):
    
    try:
        json_file = os.path.join(img_dir, "style_"+d+".json")
        with open(json_file) as f:
            imgs_anns = json.load(f)

        dataset_dict = []

        images = imgs_anns["images"] #list
        annos = imgs_anns["annotations"] #list

        for d,v in zip(images, annos): # v랑 d는 딕셔너리
    
            record = {}
            objs = []
    
    
            bbox = v["bbox"]
            poly = v["segmentation"]
            catid = v["category_id"]
            obj = {
                "bbox":bbox,
                "bbox_mode":BoxMode.XYXY_ABS,
                "segmentation":[poly],
                "category_id": catid
         }
            objs.append(obj)
    
            record["filename"] = d["file_name"]
            record["image_id"] = d["id"]
            record["height"] = d["height"]
            record["width"] = d["width"]

            record["annotation"] = objs
        
            dataset_dict.append(record)
        
        return dataset_dict

    except:
        json_file = os.path.join(img_dir)
        with open(json_file) as f:
            imgs_anns = json.load(f)
        dataset_dict = []

        images = imgs_anns["images"] #list
        annos = imgs_anns["annotations"] #list

        for d,v in zip(images, annos): # v랑 d는 딕셔너리
    
            record = {}
            objs = []
    
    
            bbox = v["bbox"]
            poly = v["segmentation"]
            catid = v["category_id"]
            obj = {
                "bbox":bbox,
                "bbox_mode":BoxMode.XYXY_ABS,
                "segmentation":[poly],
                "category_id": catid
         }
            objs.append(obj)
    
            record["filename"] = d["file_name"]
            record["image_id"] = d["id"]
            record["height"] = d["height"]
            record["width"] = d["width"]

            record["annotation"] = objs
        
            dataset_dict.append(record)
        
        return dataset_dict
        
#%% dataset register
dataset_dicts = get_dataset_dicts(jsonpath, d ="train") # 적은수에 맞게 알아서 합쳐짐 train 10141개 

'''
categories = []
for d in dataset_dicts["categories"]:
    categories.append(d["name"])'''
#%%
'''
from detectron2.structures import BoxMode

dataset_dict = []

images = dataset_dicts["images"] #list
annos = dataset_dicts["annotations"] #list

for d,v in zip(images, annos): # v랑 d는 딕셔너리
    
    record = {}
    objs = []
    
    
    bbox = v["bbox"]
    poly = v["segmentation"]
    catid = v["category_id"]
    obj = {
        "bbox":bbox,
        "bbox_mode":BoxMode.XYXY_ABS,
        "segmentation":[poly],
        "category_id": catid
    }
    objs.append(obj)
    
    record["filename"] = d["file_name"]
    record["image_id"] = d["id"]
    record["height"] = d["height"]
    record["width"] = d["width"]

    record["annotation"] = objs
        
    dataset_dict.append(record)
dataset_dict'''
#%%
for d in ["train", "val"]:
    DatasetCatalog.register("stylebot_" + d, lambda d=d: get_dataset_dicts(jsonpath,d)["images"])
    MetadataCatalog.get("stylebot_" + d).set(thing_classes= "clothes")
stylebot_metadata = MetadataCatalog.get("stylebot_train")

#%% 
#os.chdir("./stylebot/")
#%%
for d in random.sample(dataset_dicts, 5):
    #img = cv2.imread("." + d["filename"])
    img = cv2.imread(d["filename"])
    visualizer = Visualizer(img[:,:,::-1], metadata = stylebot_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    output = out.get_image()[:,:,::-1]
    rgb_im = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(20,20))
    plt.imshow(rgb_im)

#%% 7/28 to do : dataloader 만져보기 

cfg = get_cfg()
point_rend.add_pointrend_config(cfg)

#cfg.merge_from_file("/home/user/project/pytorch-unet-family/projects/UNet/Base-UNet-DS16-Semantic.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
#cfg.DATASETS.TRAIN = ("stylebot_train",) # I put my training dataset here
#cfg.DATASETS.TEST = ("stylebot_val",) # I put my validation dataset here

#cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("stylebot_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
'''
cfg.MODEL.BACKBONE.ACTIVATION = 'relu'
cfg.MODEL.BACKBONE.UNET_CHANNELS = [64, 128, 256, 512, 1024]

cfg.MODEL.SEM_SEG_HEAD.UNETPP_USE_SUBNET_IN_INFERENCE = 0
cfg.MODEL.SEM_SEG_HEAD.SEM_SEG_LOSS_TYPE = 'CrossEntropy'

cfg.INPUT.ENABLE_RANDOM_BRIGHTNESS = None
cfg.INPUT.ENABLE_RANDOM_CONTRAST = None
'''
#cfg.DATALOADER.NUM_WORKERS = 1
#cfg.SOLVER.IMS_PER_BATCH = 8
#cfg.SOLVER.BASE_LR = 0.00005 
#cfg.SOLVER.MAX_ITER = 350 
#cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
#cfg.MODEL.RETINANET.NUM_CLASSES = 35
#cfg.TEST.EVAL_PERIOD = 500
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")

# %%
#os.chdir("./.")
#os.getcwd()
'''
1. 일단 여기까지 문제가 디렉토리를 제대로 못읽어 온다는 점
2. U-net 이든 뭐든 pytorch로 코드를 짜보자

# 1. 해결해보기 실행 
json_file_t = os.path.join(jsonpath, "stylebot_train.json")

with open(json_file_t) as f:
  imgs_anns = json.load(f)

for d in imgs_anns['images']:
    d["file_name"] = d["file_name"].replace("/stylebot_wild_seg/","./stylebot_wild_seg/")

json_file_v = os.path.join(jsonpath, "stylebot_val.json")

with open(json_file_v) as f:
  imgs_anns2 = json.load(f)

for d in imgs_anns2['images']:
    d["file_name"] = d["file_name"].replace("/stylebot_wild_seg/","./stylebot_wild_seg/")

with open(jsonpath + "style_train", 'w', encoding='utf-8') as make_file:
    json.dump(imgs_anns, make_file,indent=4)
with open(jsonpath + "style_val", 'w', encoding='utf-8') as make_file:
    json.dump(imgs_anns2, make_file, indent= 4)
    '''
#%%
from detectron2.engine import DefaultTrainer
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# %%
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
cfg.DATASETS.TEST = ("stylebot_val", )
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_dataset_dicts(jsonpath, d ="val")
for d in random.sample(dataset_dicts, 3):    
    im = cv2.imread(d["filename"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=stylebot_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize = (14, 10))
    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    plt.show()
# %%
