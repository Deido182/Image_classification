import pandas
import random
import functools
import math
import abc
import os
import time

### START constants ###

TRAINING_SET = pandas.read_csv(os.path.dirname(os.path.abspath(__file__)) + "/../Training_set/fashion-mnist_train.csv")
TEST_SET = pandas.read_csv(os.path.dirname(os.path.abspath(__file__)) + "/../Test_set/fashion-mnist_test.csv")
FEATURES_TO_LOOK = 25
MAX_DEPTH = 10
COLUMNS = TRAINING_SET.shape[1]
MAX_LABEL = TRAINING_SET["label"].max()
MAX_FEATURE = functools.reduce(lambda acc, curr: max(acc, curr), TRAINING_SET.max())
TRAINING_SET_SIZE = TRAINING_SET.shape[0]
TEST_SET_SIZE = TEST_SET.shape[0]
DES_TRAIN_SUB_SIZE = (1 << (MAX_DEPTH + 1)) * math.log(2 * math.e * FEATURES_TO_LOOK * (MAX_FEATURE + 1) * (MAX_LABEL + 1))
# To bound the variance error.
# The class of tree predictors we can build with the method below 
# allows at most FEATURES_TO_LOOK * (MAX_FEATURE + 1) different tests for an 
# internal node. Morover, we have (MAX_LABEL + 1) possible choices for 
# leaves labels. Finally: the maximum number of nodes allowed is 2^(MAX_DEPTH + 1) - 1.
def get_extractions(training_set_size):
	return int(math.log(1 - DES_TRAIN_SUB_SIZE / training_set_size) / math.log(1 - 1 / training_set_size))
# However, we cannot increase it anymore (for time reasons).

### END constants ###

### START utils ###

buckets = [0 for i in range(0, MAX_FEATURE + 1)] # To speed up and save memory

class Predictor(abc.ABC):
	@abc.abstractmethod
	def predict(self, x):
		pass

def get_training_sub(training_set_size):
	if(training_set_size <= DES_TRAIN_SUB_SIZE): 
		l = [i for i in range(0, training_set_size)]
		random.shuffle(l)
		return set(l)
	return set([random.randint(0, training_set_size - 1) for i in range(0, get_extractions(training_set_size))])
	
def get_features_sub():
	features_sub = set()
	while len(features_sub) < FEATURES_TO_LOOK:
		features_sub.add(random.randint(1, COLUMNS - 1))
	return list(features_sub)
	
### END utils ###

### START loss function ###

def l(yt, yp):
	return 0 if yt == yp else 1
	
def error(predictor, DF, DF_SIZE):
	errors = 0
	for ind in range(0, DF_SIZE):
		errors += l(DF.iloc[ind]["label"], predictor.predict(DF.iloc[ind]))
	return errors / DF_SIZE
	
### END loss function ###

### START label assignment ###

def majority_label(votes):
	maj = 0
	for lab in range(1, MAX_LABEL + 1):
		if votes[lab] > votes[maj]:
			maj = lab
	return maj

def count_votes(labels):
	votes = [0 for i in range(0, MAX_LABEL + 1)]
	for lab in labels:
		votes[lab] += 1
	return votes

def get_maj_label(labels):
	return majority_label(count_votes(labels))

### END label assignment ###

### START split ###

def get_split_test(routed_to_ind, features_ind, training_set):
	random.shuffle(features_ind) # To encourage diversity and reduce bias
	best_count = 0
	best_feat_ind = None
	best_val = None
	desired_size = len(routed_to_ind) >> 1
	coeff = 9 / 10
	for feat_ind in features_ind: # It's just an heuristic of course
		 for val in range(0, len(buckets)):
			 buckets[val] = 0
		 for ind in routed_to_ind:
			 buckets[training_set.iloc[ind, feat_ind]] += 1
		 count = 0
		 for val in range(0, len(buckets)):
			 if buckets[val] == 0:
				 continue
			 count += buckets[val]
			 if abs(desired_size - count) < abs(desired_size - best_count):
				 best_count = count
				 best_feat_ind = feat_ind
				 best_val = val
		 if coeff * desired_size <= best_count and best_count <= (2 - coeff) * desired_size:
			 break # It is enough
	return None if best_count == 0 else lambda x: x[training_set.columns[best_feat_ind]] <= best_val

### END split ###

### START tree ###

class Node(Predictor):
	def __init__(self, routed_to_ind, features_ind, training_set, depth):
		self.test = None
		if depth < MAX_DEPTH:
			test = get_split_test(routed_to_ind, features_ind, training_set)
			if test != None:
				left_subset = set(filter(lambda ind: test(training_set.iloc[ind]), routed_to_ind))
				right_subset = routed_to_ind.difference(left_subset)
				self.test = test
				self.left = Node(left_subset, features_ind, training_set, depth + 1)
				self.right = Node(right_subset, features_ind, training_set, depth + 1)
		if self.test == None:
			self.label = get_maj_label(training_set.iloc[list(routed_to_ind)]["label"])
			
	def predict(self, x):
		if self.test == None:
			return self.label
		else:
			return (self.left if self.test(x) else self.right).predict(x)

def build_tree(samples_ind, features_ind, training_set):
	return Node(samples_ind, features_ind, training_set, 0)

### END tree ###

### START forest ###

class Forest(Predictor):
	def __init__(self, num_of_trees, training_set, training_set_size):
		self.trees = [build_tree(get_training_sub(training_set_size), get_features_sub(), training_set) for i in range(0, num_of_trees)]
		
	def add_tree(self, training_set, training_set_size):
		self.trees.append(build_tree(get_training_sub(training_set_size), get_features_sub(), training_set))
		
	def predict(self, x):
		return get_maj_label([tree.predict(x) for tree in self.trees])

### END forest ###

def study_trees():
	start = int(round(time.time() * 1000))	
	test_file = open(os.path.dirname(os.path.abspath(__file__)) + "/../Experiments/Test_exp", "w")
	EPS = 1e-03
	ATTEMPTS = 3
	best_test = 1.0
	curr_att = 0
	forest = Forest(0, TRAINING_SET, TRAINING_SET_SIZE)
	while True:
		forest.add_tree(TRAINING_SET, TRAINING_SET_SIZE)
		curr_test = error(forest, TEST_SET, TEST_SET_SIZE)
		if best_test - curr_test <= EPS:
			if curr_att == ATTEMPTS - 1:
				break
			curr_att += 1
		else:
			curr_att = 0
			best_test = curr_test
		test_file.write(str(len(forest.trees)) + " " + str(curr_test) + "\n")
		print("|FOREST|: " + str(len(forest.trees)) + ", TEST ERROR: " + str(curr_test))
	test_file.close()
	end = int(round(time.time() * 1000))
	print("The experiment required " + str((end - start) / 1000) + " seconds")

def study_forest2():
	start = int(round(time.time() * 1000))	
	forest2_file = open(os.path.dirname(os.path.abspath(__file__)) + "/../Experiments/Forest2_exp", "w")
	for i in range(0, 10):
		forest = Forest(1, TRAINING_SET, TRAINING_SET_SIZE)
		test1 = error(forest, TEST_SET, TEST_SET_SIZE)
		forest.add_tree(TRAINING_SET, TRAINING_SET_SIZE)
		test2 = error(forest, TEST_SET, TEST_SET_SIZE)
		forest2_file.write(str(test1 - test2) + "\n")
		print("1: " + str(test1) + ", 2: " + str(test2))
	forest2_file.close()
	end = int(round(time.time() * 1000))
	print("The experiment required " + str((end - start) / 1000) + " seconds")

def k_fold_cross_val(k, max_trees):
	start = int(round(time.time() * 1000))	
	k_fold_file = open(os.path.dirname(os.path.abspath(__file__)) + "/../Experiments/K_fold_exp", "w")
	dev_i_size = int(TRAINING_SET_SIZE / k)
	train_ind = [i for i in range(0, TRAINING_SET_SIZE)]
	for t in range(1, max_trees + 1):
		random.shuffle(train_ind)
		folds = [(train_ind[:i * dev_i_size] + train_ind[(i + 1) * dev_i_size:], train_ind[i * dev_i_size : (i + 1) * dev_i_size]) for i in range(0, k)]
		mean = 0.0
		for (s_i, d_i) in folds:
			forest = Forest(t, TRAINING_SET.iloc[s_i], len(s_i))
			mean += error(forest, TRAINING_SET.iloc[d_i], len(d_i))
		mean /= k
		k_fold_file.write(str(t) + " " + str(mean) + "\n")
		print("|FOREST|: " + str(t) + ", TEST ERROR: " + str(mean))
	k_fold_file.close()
	end = int(round(time.time() * 1000))
	print("The experiment required " + str((end - start) / 1000) + " seconds")

def test_vs_train(forest_size):
	STEP = 10000
	start = int(round(time.time() * 1000))	
	test_vs_train_file = open(os.path.dirname(os.path.abspath(__file__)) + "/../Experiments/Test_vs_train_exp", "w")
	train_ind = [i for i in range(0, TRAINING_SET_SIZE)]
	for i in range(1, 7):
		random.shuffle(train_ind)
		Si = TRAINING_SET.iloc[train_ind[:i * STEP]]
		forest = Forest(forest_size, Si, i * STEP)
		test_error = error(forest, TEST_SET, TEST_SET_SIZE)
		train_error = error(forest, Si, i * STEP)
		test_vs_train_file.write(str(i * STEP) + " " + str(test_error) + " " + str(train_error) + "\n")
		print("|Si|: " + str(i * STEP) + ", TEST ERROR: " + str(test_error) + ", TRAIN ERROR: " + str(train_error))
	test_vs_train_file.close()
	end = int(round(time.time() * 1000))
	print("The experiment required " + str((end - start) / 1000) + " seconds")

#study_trees()
#study_forest2()
#k_fold_cross_val(5, 10)
#test_vs_train(20)

		

	

	

