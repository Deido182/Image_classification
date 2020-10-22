import matplotlib.pyplot as plt
import os

test_file = open(os.path.dirname(os.path.abspath(__file__)) + "/../Experiments/Test_exp", "r")

x = []
y_test = []

for line in test_file.readlines():
	x_i, y_test_i = line.split()
	x.append(float(x_i))
	y_test.append(float(y_test_i))

test_file.close()

'''
plt.plot(x, y_test, '.-r')
plt.xlim(0, x[len(x) - 1] + 1)
plt.ylim(0, 1)
plt.yticks([y / 100 for y in range(0, 101, 5)])
plt.xlabel("Trees")
plt.ylabel("Test error")
'''
'''
forest2_file = open(os.path.dirname(os.path.abspath(__file__)) + "/../Experiments/Forest2_exp", "r")

data = [float(line) for line in forest2_file.readlines()]

forest2_file.close()

plt.boxplot(data)
plt.ylabel("err(1) - err(2)")
plt.xticks([])
'''
'''
k_fold_file = open(os.path.dirname(os.path.abspath(__file__)) + "/../Experiments/K_fold_exp", "r")

x = []
y_k_fold = []

for line in k_fold_file.readlines():
	x_i, y_k_fold_i = line.split()
	x.append(float(x_i))
	y_k_fold.append(float(y_k_fold_i))

k_fold_file.close()

plt.plot(x[:10], y_test[:10], '.-r', label = "Over test set")
plt.plot(x[:10], y_k_fold[:10], '.-b', label = "Over 5-fold cross validation")
plt.xlim(0, 10 + 1)
plt.ylim(0, 1)
plt.yticks([y / 100 for y in range(0, 101, 5)])
plt.xlabel("Trees")
plt.ylabel("Error")
plt.legend()
'''
test_vs_train_file = open(os.path.dirname(os.path.abspath(__file__)) + "/../Experiments/Test_vs_train_exp", "r")

x = []
y_test = []
y_train = []

for line in test_vs_train_file.readlines():
	x_i, y_test_i, y_train_i = line.split()
	x.append(float(x_i))
	y_test.append(float(y_test_i))
	y_train.append(float(y_train_i))

test_vs_train_file.close()

plt.plot(x, y_test, '.-r', label = "Test")
plt.plot(x, y_train, '.-b', label = "Train")
plt.xlim(0, 70000)
plt.ylim(0, 1)
plt.yticks([y / 100 for y in range(0, 101, 5)])
plt.xlabel("Training set size")
plt.ylabel("Error")
plt.legend()
plt.show()
