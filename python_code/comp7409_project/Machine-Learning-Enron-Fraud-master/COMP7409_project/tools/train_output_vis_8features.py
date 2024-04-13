
import matplotlib.pyplot as plt


# 8 features
min_samples_split = [3, 4, 5, 8,  10]
accuracy = [0.81886, 0.81564, 0.81693, 0.81114, 0.81664]
precision = [0.35078, 0.33780, 0.33850, 0.29543, 0.31674]
recall = [0.31500, 0.30250, 0.29500, 0.23250, 0.24500]
f1 = [0.33193, 0.31918, 0.31526, 0.26021, 0.27629]



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


plt.title(' train_dataset  (8 features)  ')
plt.title(' train_dataset  (poi +7 features)  ')

plt.xlabel('min_samples_split')
plt.ylabel('Score')
plt.legend()
plt.xticks(min_samples_split)

plt.xlim([1, 12])

plt.show()