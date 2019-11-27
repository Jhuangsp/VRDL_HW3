import numpy as np
import cv2
import os
import pandas as pd
import h5py
import glob

train_folder = "./train"
test_folder = "./test"

def collapse_col(row):
    global resize_size
    new_row = {}
    new_row['img_name'] = list(row['img_name'])[0]
    new_row['labels'] = row['label'].astype(np.str).str.cat(sep='_')
    new_row['top'] = max(int(row['top'].min()),0)
    new_row['left'] = max(int(row['left'].min()),0)
    new_row['bottom'] = int(row['bottom'].max())
    new_row['right'] = int(row['right'].max())
    new_row['width'] = int(new_row['right'] - new_row['left'])
    new_row['height'] = int(new_row['bottom'] - new_row['top'])
    new_row['num_digits'] = len(row['label'].values)
    return pd.Series(new_row,index=None)

def image_data_constuctor(img_folder, img_bbox_data):
    print('image data construction starting...')
    imgs = []
    for img_file in os.listdir(img_folder):
        if img_file.endswith('.png'):
            imgs.append([img_file,cv2.imread(os.path.join(img_folder,img_file))])
    img_data = pd.DataFrame([],columns=['img_name','img_height','img_width','img','cut_img'])
    print('finished loading images...starting image processing...')
    for img_info in imgs:
        row = img_bbox_data[img_bbox_data['img_name']==img_info[0]]
        full_img = img_info[1] #cv2.normalize(cv2.cvtColor(cv2.resize(img_info[1],resize_size), cv2.COLOR_BGR2GRAY).astype(np.float64), 0, 1, cv2.NORM_MINMAX)
        cut_img = full_img.copy()[int(row['top']):int(row['top']+row['height']),int(row['left']):int(row['left']+row['width']),...]
        row_dict = {'img_name':[img_info[0]],'img_height':[img_info[1].shape[0]],'img_width':[img_info[1].shape[1]],'img':[full_img],'cut_img':[cut_img]}
        img_data = pd.concat([img_data,pd.DataFrame.from_dict(row_dict,orient = 'columns')])
    print('finished image processing...')
    return img_data

def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])

def get_bbox(index, hdf5_data):
    attrs = {}
    item = hdf5_data['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[item][key]
        values = [hdf5_data[attr.value[i].item()].value[0][0]
                  for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
        attrs[key] = values
    return attrs

def img_boundingbox_data_constructor(mat_file):
    f = h5py.File(mat_file,'r') 
    all_rows = []
    print('image bounding box data construction starting...')
    bbox_df = pd.DataFrame([],columns=['height','img_name','label','left','top','width'])
    for j in range(f['/digitStruct/bbox'].shape[0]):
        img_name = get_name(j, f)
        row_dict = get_bbox(j, f)
        row_dict['img_name'] = img_name
        all_rows.append(row_dict)
        bbox_df = pd.concat([bbox_df,pd.DataFrame.from_dict(row_dict,orient = 'columns')])
        if j % 1000 == 0:
            print('{}/{}'.format(j, f['/digitStruct/bbox'].shape[0]))
    bbox_df['bottom'] = bbox_df['top']+bbox_df['height']
    bbox_df['right'] = bbox_df['left']+bbox_df['width']
    return bbox_df

img_folder = train_folder
mat_file_name = 'digitStruct.mat'

img_bbox_data = img_boundingbox_data_constructor(os.path.join(img_folder,mat_file_name))
print(img_bbox_data[:10])

img_bbox_data_grouped = img_bbox_data.groupby('img_name')

namelist = glob.glob('train\\*.png')
namelist = list(map(lambda x: x.split('\\')[-1], namelist))

for name in namelist:
    with open('labels/trainlabels/' + name.split('.')[0] + '.txt', 'w') as out_file:
        h,w,_ = cv2.imread('./train/'+name).shape
        for idx, line in img_bbox_data_grouped.get_group(name).iterrows():
            label = int(line['label']) if int(line['label']) != 10 else 0
            x_t = min((line['left']+line['right'])/(2*w), 1.0)
            y_t = min((line['top']+line['bottom'])/(2*h), 1.0)
            w_t = min(line['width']/w, 1.0)
            h_t = min(line['height']/h, 1.0)
            out_file.write('{} {} {} {} {}\n'.format(label, x_t, y_t, w_t, h_t))
            assert x_t <= 1 and y_t <= 1 and w_t <= 1 and h_t <= 1, '{}: {}'.format(name, line)

with open('labels/train_list.txt', 'w') as out_file:
    for name in namelist:
        out_file.write('../mydata/images/train/' + name + '\n')