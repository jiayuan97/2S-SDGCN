import argparse
import pickle
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score

parser = argparse.ArgumentParser()
# parser.add_argument('--datasets', default='ntu120/xsub', choices={'kinetics', 'ntu/xsub', 'ntu/xview'},
#                     help='the work folder for storing results')
parser.add_argument('--alpha', default=0.5, help='weighted summation')
arg = parser.parse_args()

# def calculate_auc(targets, predictions):
#     """
#     Calculate performance measures on test dataset.
#     :param targets: Target vector to predict.
#     :param predictions: Predictions vector.
#     :param edges: Edges dictionary with number of edges etc.
#     :return auc: AUC value.
#     :return f1: F1-score.
#     """
#     # targets = [0 if target == 1 else 1 for target in targets]
#     auc = roc_auc_score(targets, predictions)
#     pred = [1 if p > 0.5 else 0 for p in predictions]
#     f1 = f1_score(targets, pred)
#     pos_ratio = sum(pred)/len(pred)
#     return auc, f1, pos_ratio

# dataset = arg.datasets
# label = open('/home/home/Desktop/Python/data/' + dataset + '/val_label.pkl', 'rb')
# label = open('/home/home/Desktop/Python/data/ntu120/xsetup/val_label.pkl', 'rb')
# label = np.array(pickle.load(label))
label = open('/home/home/Desktop/Code/SGCN/Slashdot_target.pkl', 'rb')
label = pickle.load(label)
# label = [0 if t == 1 else 1 for t in label]
# label = np.array(list(enumerate(pickle.load(label))))
# label=label.T
# ind = np.argsort(label[0])
# # print(ind
# label = label[:, ind]

r1 = open('/home/home/Desktop/Code/SGCN/Slashdot_prediction1.pkl', 'rb')
r11 = pickle.load(r1)
# r11 = np.array(list(enumerate(pickle.load(r1))))
auc1 = roc_auc_score(label, r11)
pred1 = [1 if p > 0.5 else 0 for p in r11]
f11 = f1_score(label, pred1)
# # r1 = open('./work_dir/' + dataset + '/sgcn_test_joint/epoch1_test_score.pkl', 'rb')
# # r1 = open('./work_dir/ntu120/cset/sgcn_test_joint/epoch1_test_score.pkl', 'rb')
# r1 = list(enumerate(pickle.load(r1)))
# # print(r1[0][1])
# # exit(0)
# r1 = sorted(r1, key=lambda x: x[0])
#
# # r2 = open('./work_dir/' + dataset + '/sgcn_test_bone/epoch1_test_score.pkl', 'rb')
r2 = open('/home/home/Desktop/Code/SGCN/Slashdot_prediction2.pkl', 'rb')
r22 = pickle.load(r2)
auc2 = roc_auc_score(label, r22)
pred2 = [1 if p > 0.5 else 0 for p in r22]
f12 = f1_score(label, pred2)
r = r11*(1-arg.alpha) + r22 * arg.alpha
auc = roc_auc_score(label, r)
pred = [1 if p > 0.5 else 0 for p in r]
f = f1_score(label, pred)
# r2 = list(enumerate(pickle.load(r2)))
# r2 = sorted(r2, key=lambda x: x[0])
# # print(label)
# # exit(0)
# # print(r1)
# # print(r2)
# right_num = right_num1 = right_num2 = total_num = right_num_5 = 0
# for i in tqdm(range(len(label[0]))):
#     _, l = label[:, i]
#     _, r11 = r1[i]
#     _, r22 = r2[i]
#     r = r11*(1-arg.alpha) + r22 * arg.alpha
#     # print(r)
#     #print(arg.alpha)
#     # rank_5 = r.argsort()[-5:]
#     # right_num_5 += int(int(l) in rank_5)
#     #print(r11)
#     # r11 = np.argmax((torch.tensor(r11).numpy()))
#     # r22 = np.argmax((torch.tensor(r22).numpy()))
#     # print(r11)
#     # print(r22)
#     # auc = roc_auc_score(label, r11)
#     right_num1 += int(l == round(r11))
#     right_num2 += int(l == round(r22))
#     #right_num2 += int(r22 == int(l))
#     # r = np.argmax((torch.tensor(r).numpy()))
#     right_num += int(l == round(r))
#     total_num += 1
# label=np.array(label)
# r11=np.array(r1)
# r22=np.array(r2)
# print(label)
# print(r11)
# print(r22)
# print(right_num1)
# print(right_num2)
# acc1 = right_num1 / total_num
# acc2 = right_num2 / total_num
# acc = right_num / total_num
# print(total_num)
# # auc = roc_auc_score(label, r11)
# # acc5 = right_num_5 / total_num
# print(acc, acc1, acc2)
# print(arg.alpha)
print(auc1, auc2, auc, f11, f12, f)


