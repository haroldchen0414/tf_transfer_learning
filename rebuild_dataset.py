# -*- coding: utf-8 -*-
# author: haroldchen0414

from imutils import paths
import shutil
import os

"""
将数据集目录变成下面这种形式, 即train-classes-image形式
|
| Food-5K_rebuild
|    |training
|    |    |food
|    |    |non-food
|    |evaluation
|    |    |food
|    |    |non-food
|    |validation
|    |    |food
|    |    |non-food
|
"""
basePath = os.path.sep.join(["E:", "dataset"])
origDataset = "Food-5K"
newDataset = "Food-5K_rebuild"
datasetSplits = ["training", "evaluation", "validation"]
classes = ["non-food", "food"]

imagePaths = list(paths.list_images(os.path.sep.join([basePath, origDataset])))

for split in datasetSplits:
    for c in classes:
        if not os.path.exists(os.path.sep.join([basePath, newDataset, split, c])):
            os.makedirs(os.path.sep.join([basePath, newDataset, split, c]))

for (i, imagePath) in enumerate(imagePaths):
    if i % 1000 == 0:
        print("正在处理第{}张图片, 剩余{}张...".format(i, len(imagePaths) - i))

    ds = imagePath.split(os.path.sep)[-2]
    c = os.path.basename(imagePath).split("_")[0]
    shutil.copy2(imagePath, os.path.sep.join([basePath, newDataset, ds, classes[int(c)]]))
