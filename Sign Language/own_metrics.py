from sklearn import metrics
import numpy as np

def specificity(y_true, y_pred, labels=None, average=None, sample_weight=None):
	average_options = (None, 'macro', 'weighted')
	if average not in average_options:
		raise ValueError('average has to be one of ' + str(average_options))
	
	'''#Find classes present in truth and prediction
	present_labels, counts = np.unique(y_true + y_pred, return_counts=True)
	
	#If sample_weight is None, use true positives as
	
	
	
	if labels is not None:
		n_labels = len(labels)
		labels = sorted(np.hstack([labels, np.setdiff1d(present_labels, labels, assume_unique=True)]).tolist())
		
		overall_label_counts = np.zeros(len(labels))
		for l in labels:
			if l in present_labels:
				overall_label_counts[labels.index(l)]=counts[present_labels.tolist().index(l)]
	
	#Find how many classes are if labels = None
	'''
	
	n_labels = len(list(set(y_true+y_pred)))
	spec = [0]*n_labels
	
	conf_m = metrics.confusion_matrix(y_true,y_pred)
	
	for i in range(n_labels):
		tp = conf_m[i, i]
		fp = conf_m[:, i].sum() - conf_m[i, i]
		fn = conf_m[i, :].sum() - conf_m[i, i]
		tn = conf_m.sum() - tp - fp - fn
		spec[i]=tn/(tn+fp)
	
	if average == 'macro':
		spec = np.average(spec)

	return spec
	
''' macro returns unweighted mean of each class. Doesn´t count for imbalance in the data
	micro returns metrics calculated globally (not class discrimined)
	weighted returns macro metric but accounts for imbalance
'''

	
	
y1 = [0,1,2,3,4]
y2 = [1,1,2,2,5]
a = metrics.confusion_matrix(y1,y2)
sp = specificity(y1,y2)

print(a)
print(sp)