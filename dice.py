# -*- coding: utf-8 -*-
#版权归属lv总
import os
import sys
import glob
import cv2
import numpy as np
import nibabel as nib
import pandas as pd

from sklearn.metrics import confusion_matrix

name_list = ['femur-left', 'femur-right',
             'hip bone-left', 'hip bone-right', 'iliacus-left', 'iliacus-right',
             'muscle of the gluteus maximus-left', 'muscle of the gluteus maximus-right',
             'muscle of the gluteus medius-left', 'muscle of the gluteus medius-right',
             'muscle of the gluteus minimus-left', 'muscle of the gluteus minimus-right',
             'Muscle of the obturator internus-left', 'Muscle of the obturator internus-right',
             'pectineus-left', 'pectineus-right',
             'piriformis-left', 'piriformis-right', 'psoas major-left', 'psoas major-right',
             'quadratus femoris-left', 'quadratus femoris-right', 'rectus abdominis-left', 'rectus abdominis-right',
             'rectus femoris-left', 'rectus femoris-right', 
             'Sartorius-left', 'Sartorius-right']

def fileList(imgpath, filetype):
	return glob.glob(imgpath + filetype)

def analyze_name(path):
    name = os.path.split(path)[1]
    name = os.path.splitext(name)[0]
    return name

def mkdir_if_not_exist(dir_name, is_delete=False):
    try:
        if is_delete:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
                print(u'[INFO] Dir "%s" exists, deleting.' % dir_name)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(u'[INFO] Dir "%s" not exists, creating.' % dir_name)
        return True
    except Exception as e:
        print('[Exception] %s' % e)
        return False

def collect(gtPath, predPath):
    gtlist = fileList(gtPath, '*')
    predlist = fileList(predPath, '*')
    # check if two folders has the same numbers of images
    if len(gtlist) != len(predlist):
        return 1, None, None

    gtlist.sort()
    predlist.sort()
    gt, pred = [], []

    for i in range(len(gtlist)):
        # check if two images have the same filenames
        if analyze_name(gtlist[i]) != analyze_name(predlist[i]):
            return 2, None, None

        # open .nii label
        predImg = nib.load(predlist[i]).get_fdata()
        gtImg = nib.load(gtlist[i]).get_fdata()       

        # check shape
        if gtImg.shape != predImg.shape:
            return 3, None, None
        
        gt.append(gtImg)
        pred.append(predImg)

    return 0, gt, pred, gtlist

def evaluate(gt, pred, name, csvpath):
    mkdir_if_not_exist('./result/'+csvpath)
    df_avg = pd.DataFrame({'accuracy':[None]*len(gt),
                       'sensitivity':[None]*len(gt),
                       'specificity':[None]*len(gt),
                       'dice coefficient similarity':[None]*len(gt)}, index=[analyze_name(x) for x in name])
    total_acc, total_sen, total_spec, total_dice = 0, 0, 0, 0

    for i in range(len(gt)):
        print('[INFO] Evaluating ' + analyze_name(name[i]))
        classes = int(np.max(gt[i]))
        avg_acc, avg_sen, avg_spec, avg_dice = 0, 0, 0, 0
        df = pd.DataFrame({'accuracy':[None]*classes,
                       'sensitivity':[None]*classes,
                       'specificity':[None]*classes,
                       'dice coefficient similarity':[None]*classes}, index=name_list)

        for j in range(1, classes+1):
            print('Evaluating ' + name_list[j-1])
            a = gt[i].copy().astype(np.uint8)
            b = pred[i].copy().astype(np.uint8)
            a[np.where(a!=j)] = 0
            a[np.where(a==j)] = 1
            b[np.where(b!=j)] = 0
            b[np.where(b==j)] = 1
            a = np.array(a, np.uint8).flatten()
            b = np.array(b, np.uint8).flatten()
            confusion = confusion_matrix(a, b)
            # tp, tn, fp, fn
            tn = confusion[0, 0]
            tp = confusion[1, 1]
            fn = confusion[1, 0]
            fp = confusion[0, 1]
            # metrics, in case of ill-defined
            if fp + fn == 0:
                sen = 1
                dice = 1
            else:
                sen = tp/(tp + fn)
                dice = 2*tp/(2*tp + fp + fn)
            if tn + fp == 0:
                spec = 1
            else:
                spec = tn/(tn + fp)
            if tn + fn == 0:
                acc = 1
            else:
                acc = (tp + tn)/(tp + tn + fp + fn)

            df['accuracy'][name_list[j-1]] = acc
            avg_acc += acc
            df['sensitivity'][name_list[j-1]] = sen
            avg_sen += sen
            df['specificity'][name_list[j-1]] = spec
            avg_spec += spec
            df['dice coefficient similarity'][name_list[j-1]] = dice
            avg_dice += dice

            print(acc, sen, spec, dice)

        df.to_csv(os.path.join('./result',csvpath, analyze_name(name[i])+'.csv'), encoding='utf-8')

        df_avg['accuracy'][analyze_name(name[i])] = (avg_acc / classes)
        total_acc += (avg_acc / classes)
        df_avg['sensitivity'][analyze_name(name[i])] = (avg_sen / classes)
        total_sen += (avg_sen / classes)
        df_avg['specificity'][analyze_name(name[i])] = (avg_spec / classes)
        total_spec += (avg_spec / classes)
        df_avg['dice coefficient similarity'][analyze_name(name[i])] = (avg_dice / classes)
        total_dice += (avg_dice / classes)

    df_avg.to_csv(os.path.join('./result',csvpath, 'average.csv'), encoding='utf-8')

    return total_acc / len(gt), total_sen / len(gt), total_spec / len(gt), total_dice / len(gt)

if __name__ == '__main__':
    gtPath = 'D:/desktop/isicdm/ISICDM dataset/dice-test/gt2/'
    predPath = 'D:/desktop/isicdm/ISICDM dataset/dice-test/pred2/'
    stat, gt, pred, name = collect(gtPath, predPath)
    if stat == 0:
        print(evaluate(gt, pred, name, 'sustc'))
    else:
        print('Error')
