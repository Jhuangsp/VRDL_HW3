import cv2
import json
import os
import glob
import pprint

pp = pprint.PrettyPrinter(indent=4)

filepath = '.\\output'
images = glob.glob(filepath + '\\*.png')
answers = list(map(lambda i: i + '.txt', images))
answers = list(
    map(lambda i: [int(i.split('\\')[-1].split('.')[0]), i], answers))
answers.sort()
answers = [i for _, i in answers]

my_answer = []
for ans in answers:
    tmp = {'bbox': [], 'label': [], 'score': []}
    if os.path.isfile(ans):
        with open(ans, 'r') as f:
            for line in f.readlines():
                x1, y1, x2, y2, cls, conf = line[:-2].split(' ')
                tmp['bbox'].append([int(y1), int(x1), int(y2), int(x2)])
                tmp['label'].append(int(cls) if int(cls) != 0 else 10)
                tmp['score'].append(float(conf))
    my_answer.append(tmp)
