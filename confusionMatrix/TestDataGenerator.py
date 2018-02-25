#!/usr/bin/python
import os
import random
import shutil

shutil.rmtree("test_data_20", ignore_errors=True)
os.makedirs("test_data_20")
os.makedirs("test_data_20/val")
os.makedirs("test_data_20/train")
for f in random.sample(os.listdir("data/val"), 20):
#	Copy 20 random classes from val and train to the newly created directories
    shutil.copytree("data/val/" + f, "test_data_20/val/"+f)
    shutil.copytree("data/train/" + f, "test_data_20/train/"+f)
