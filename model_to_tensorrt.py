# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 11:05:10 2025

@author: user
"""

from ultralytics import YOLO

# 1) 학습된 .pt 모델 로드
model = YOLO('./best.pt')
model.export(format="engine", imgsz=640, half=True, device=0)