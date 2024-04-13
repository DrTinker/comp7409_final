
# train
# min_samples_split=2 Accuracy: 0.83408	Precision: 0.50244	Recall: 0.46300	F1: 0.48192
# min_samples_split=3 Accuracy: 0.85558	Precision: 0.58487	Recall: 0.46000	F1: 0.51497
# min_samples_split=4 Accuracy: 0.85717	Precision: 0.59202	Recall: 0.46000	F1: 0.51773
# min_samples_split=5 Accuracy: 0.85833	Precision: 0.59778	Recall: 0.45850	F1: 0.51896
# min_samples_split=6 Accuracy: 0.86283	Precision: 0.62041	Recall: 0.45600	F1: 0.52565
# min_samples_split=7 Accuracy: 0.86867	Precision: 0.65252	Recall: 0.45350	F1: 0.53510
# min_samples_split=8 Accuracy: 0.86842	Precision: 0.65046	Recall: 0.45500	F1: 0.53545
# min_samples_split=9 Accuracy: 0.86933	Precision: 0.65813	Recall: 0.44950	F1: 0.53417
# min_samples_split=10Accuracy: 0.87058	Precision: 0.67048	Recall: 0.43950	F1: 0.53096
# min_samples_split=15Accuracy: 0.87783	Precision: 0.73882	Recall: 0.41300	F1: 0.52983
# min_samples_split=20Accuracy: 0.86450	Precision: 0.71445	Recall: 0.31150	F1: 0.43384

import matplotlib.pyplot as plt

# train aran 2 features
min_samples_split = [3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
accuracy = [ 0.85558, 0.85717, 0.85833, 0.86283, 0.86867, 0.86842, 0.86933, 0.87058, 0.87783, 0.86450]
precision = [ 0.58487, 0.59202, 0.59778, 0.62041, 0.65252, 0.65046, 0.65813, 0.67048, 0.73882, 0.71445]
recall = [ 0.46000, 0.46000, 0.45850, 0.45600, 0.45350, 0.45500, 0.44950, 0.43950, 0.41300, 0.31150]
f1 = [ 0.51497, 0.51773, 0.51896, 0.52565, 0.53510, 0.53545, 0.53417, 0.53096, 0.52983, 0.43384]


plt.plot(min_samples_split, accuracy, label='Accuracy')
plt.plot(min_samples_split, precision, label='Precision')
plt.plot(min_samples_split, recall, label='Recall')
plt.plot(min_samples_split, f1, label='F1')
#
max_accuracy = max(accuracy)
max_precision = max(precision)
max_recall = max(recall)
max_f1 = max(f1)

plt.annotate(f'{max_accuracy:.4f}', (min_samples_split[accuracy.index(max_accuracy)], max_accuracy), textcoords='offset points', xytext=(0,5), ha='center')
plt.annotate(f'{max_precision:.4f}', (min_samples_split[precision.index(max_precision)], max_precision), textcoords='offset points', xytext=(0,5), ha='center')
plt.annotate(f'{max_recall:.4f}', (min_samples_split[recall.index(max_recall)], max_recall), textcoords='offset points', xytext=(0,5), ha='center')
plt.annotate(f'{max_f1:.4f}', (min_samples_split[f1.index(max_f1)], max_f1), textcoords='offset points', xytext=(0,5), ha='center')
# 显示最大值对应的数据
plt.xlim([min_samples_split[0], min_samples_split[-1]])
plt.ylim([0, 1])

# 2 features
plt.title(' train_dataset  (poi, exercised_stock_options , poi_email_ratio)  ')
# plt.title(' train_dataset  (poi + 7 features)  ')

plt.xlabel('min_samples_split')
plt.ylabel('Score')
plt.legend()
plt.xticks(min_samples_split)

plt.xlim([1, 21])

plt.show()