import numpy as np
import matplotlib.pyplot as plt
import torch
from pytope import Polytope
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# read ./results/42898570_boxes.pkl
import pickle
with open('results/r_boxes.pkl', 'rb') as f:
    boxes = pickle.load(f)
boxes = boxes[0]

box_objs = []
for box in boxes:
    id, points, label = box
    poly = (id, Polytope(points), label)
    print(f"Box ID: {id}, Label: {label}")
    box_objs.append(poly)

# read the point cloud
pcd_data = torch.load('results/r_xyzrgb.pt')  # [N,6]
xyzrgbd = pcd_data.numpy()

# get points inside a box
def points_in_box(xyz, box_poly):
    Ab = box_poly.get_H_rep()
    A, b = Ab[0], Ab[1]
    in_box_mask = np.all(np.dot(xyz, A.T) <= np.repeat(b, xyz.shape[0], axis=1).T + 1e-6, axis=1)
    return xyz[in_box_mask]
# import pdb;pdb.set_trace()
for box_id, box_poly, label in box_objs:
    # import pdb;pdb.set_trace()
    in_box_points = points_in_box(xyzrgbd[:,:3], box_poly)
    print(f"Box ID: {box_id}, Label: {label}, Points inside: {in_box_points.shape[0]}")
    # if label=='cup':
    # label.replace('/','')
    if label=='Orange/Tangerine':
        label='Orange'
    # import pdb;pdb.set_trace()
    torch.save(torch.from_numpy(in_box_points), f'./results/{label}_obj.pt')


# save the point cloud for generating grasps


# load text features of CLIP
# text_features = torch.load('data/class_features.pt')
# print(text_features.shape)