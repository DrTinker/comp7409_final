# val
# min_samples_split=3 Accuracy: 0.85675	Precision: 0.58978	Recall: 0.46150	F1: 0.51781
# min_samples_split=4 Accuracy: 0.85783	Precision: 0.59545	Recall: 0.45850	F1: 0.51808
# min_samples_split=5 Accuracy: 0.85850	Precision: 0.59818	Recall: 0.46000	F1: 0.52007
# min_samples_split=6 Accuracy: 0.86400	Precision: 0.62500	Recall: 0.46000	F1: 0.52995
# min_samples_split=7 Accuracy: 0.86867	Precision: 0.65274	Recall: 0.45300	F1: 0.53483
# min_samples_split=8 Accuracy: 0.86867	Precision: 0.65318	Recall: 0.45200	F1: 0.53428
# min_samples_split=9 Accuracy: 0.86908	Precision: 0.65807	Recall: 0.44650	F1: 0.53202
# min_samples_split=10 Accuracy: 0.87108	Precision: 0.67303	Recall: 0.44050	F1: 0.53249
# min_samples_split=15 Accuracy: 0.87800	Precision: 0.74101	Recall: 0.41200	F1: 0.52956
# min_samples_split=20Accuracy: 0.86450	Precision: 0.71445	Recall: 0.31150	F1: 0.43384
#

import matplotlib.pyplot as plt

# vel dataset
min_samples_split = [3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
accuracy = [0.85675, 0.85783, 0.85850, 0.86400, 0.86867, 0.86867, 0.86908, 0.87108, 0.87800, 0.86450]
precision = [0.58978, 0.59545, 0.59818, 0.62500, 0.65274, 0.65318, 0.65807, 0.67303, 0.74101, 0.71445]
recall = [0.46150, 0.45850, 0.46000, 0.46000, 0.45300, 0.45200, 0.44650, 0.44050, 0.41200, 0.31150]
f1 = [0.51781, 0.51808, 0.52007, 0.52995, 0.53483, 0.53428, 0.53202, 0.53249, 0.52956, 0.43384]

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

plt.title('val_dataset')
plt.xlabel('min_samples_split')
plt.ylabel('Score')
plt.legend()
plt.xticks(min_samples_split)
plt.xlim([1, 21])

plt.show()