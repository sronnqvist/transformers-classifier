import os
import numpy as np
from sklearn.metrics import classification_report

PATH="output/"

for lang in "fi fr sv".split():
    f1s = []
    files = [f for f in os.listdir(PATH) if f.startswith('xlmrL2-fi-%s-' % lang) and f.endswith('gold.npy')]
    files=["xlmrL2-fi-fi-lr2e-5-ep4-1.h5-epoch3.gold.npy"]
    for filename in files:
        filename = PATH+filename
        print(filename)
        gold = np.load(filename)
        pred = np.load(filename.replace('gold','preds'))
        #labels = np.load(filename.replace('gold','class_labels'),allow_pickle=True)
        labels_fr=['HI', 'ID', 'IN', 'IP', 'LY', 'NA', 'OP', 'SP']
        for i in range(gold.shape[0]):
            if sum(gold[i]) > 1:
                gold[i] *= 0
        #print(labels)
        print(classification_report(gold, pred))
        #classification_report(gold, pred, output_dict=True)[str(list(labels).index('IP'))]['f1-s
        #print(classification_report(gold, pred, output_dict=True)[str(list(labels).index('IP'))]['f1-score'])
        f1s.append(classification_report(gold,pred,output_dict=True)['micro avg']['f1-score'])
        print()
    print(lang, "f1", np.mean(f1s))


