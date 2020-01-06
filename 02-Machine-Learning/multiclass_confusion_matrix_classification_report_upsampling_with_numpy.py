# return comprehensive metrics (confustion matrix + classification report) for multiple classes
# implemented entirely with numpy (no scikit-learn)
def metrics(actual, predicted):
    
    actual = np.ravel(actual)
    predicted = np.ravel(predicted)
    labels = np.unique(np.append(actual, predicted, axis=None))
    assert actual.shape == predicted.shape, 'Different number of actual and predicted labels'
    assert labels.shape[0] > 1,             'Provide at lease two classes'
        
    # confusion matrix
    print('CV Accuracy', round(((actual == predicted).sum() / len(predicted)), 4))
    imap = {item: idx for idx, item in enumerate(labels)}
    cm   = np.zeros((labels.shape[0], labels.shape[0]))
    for a, p in zip(actual, predicted):
        cm[imap[a], imap[p]] += 1
    print('\nconfusion matrix\n', cm)
        
    # precision, recall, f1-score, support
    recalls = np.diag(cm) / np.sum(cm, axis = 1)
    precisions = np.diag(cm) / np.sum(cm, axis = 0)
    f1scores = np.divide(np.multiply(precisions, recalls)*2, np.add(precisions, recalls))
    supports = np.sum(cm, axis=1)

    print('\n{:>11}{:>11}{:>10}{:>11}{:>10}\n'.format('', 'precision', 'recall', 'f1-score', 'support'))
    for i in zip(labels, precisions, recalls, f1scores, supports):
        print('{:>11}{:>11.4f}{:>10.4f}{:>11.4f}{:>10.0f}'.format(*i))
    print('\n{:>11}{:>11.4f}{:>10.4f}{:>11.4f}{:>10.0f}\n'.format('avg / total', np.mean(precisions), np.mean(recalls),
                                                             np.mean(f1scores), np.sum(supports)))
               
    return cm


# upsample one array with multiple classes
# implemented entirely with numpy
def upsample(arr):
    a = np.copy(arr)
    unq, unq_idx, unq_cnt = np.unique(a[:, -1], return_inverse=True, return_counts=True)    # get classes, indices, counts
    cnt = np.max(unq_cnt)                                                                   # majority class count
    res = np.empty((cnt*len(unq) - len(a),) + a.shape[1:], a.dtype)                         # to store upsamples data points
    slices = np.concatenate(([0], np.cumsum(cnt - unq_cnt)))                                # upsampling counts
    for j in range(len(unq)):
        indices = np.random.choice(np.where(unq_idx==j)[0], cnt - unq_cnt[j])               # get the difference by index
        res[slices[j]:slices[j+1]] = a[indices]
    res = np.vstack((a, res))                                                               # concat class and difference
    return res                                                                              # to ensure all data points are used


# f(x) to handle upsampling for multiple classes implemented with numpy
def cv_upsampled(model, X, Y, cv=100):
        
    X_copy, Y_copy = np.copy(X), np.copy(Y)
    c = np.concatenate((X_copy, Y_copy.reshape(-1,1)), axis=1)
    scores, all_ytrue, all_ypred = [], [], []
    
    for i in range(cv):
    
        # get classes and counts
        np.random.shuffle(c)
        unq, unq_cnt = np.unique(c[:, -1], return_counts=True)
        
        # make sure each class has at least 5 samples
        for pair in zip(unq_cnt, unq):
            assert pair[0] > 3, 'Number of samples for class {} must be at least 4'.format(pair[1])

        # arrays for each class
        arrs = []
        for j in unq:
            arrs.append(c[c[:, -1] == j])

        # split array for each class into train and test sets
        arrs_train, arrs_test = [],[]
        for j in range(len(unq)):
            if unq_cnt[j] >= 5 and unq_cnt[j] <= 7:
                train, test = np.split(arrs[j], [int(0.6 * len(arrs[j]))])
            else:
                train, test = np.split(arrs[j], [int(0.8 * len(arrs[j]))])
            arrs_train.append(train)
            arrs_test.append(test)
            if i % 25 == 0:
                print('Iteration {}: shape of class {} is {} for training and {} for testing'.format(i, unq[j], train.shape, test.shape))
    
        # merge train and test sets
        train_joint, test_joint = np.vstack(arrs_train), np.vstack(arrs_test)
                       
        # upsample train and test samples separately
        train_upsampled, test_upsampled = upsample(train_joint), upsample(test_joint)         
        np.random.shuffle(train_upsampled)
        np.random.shuffle(test_upsampled)
                
        # classify on upsampled train and test sets 
        train_X, train_y= train_upsampled[:, :-1], train_upsampled[:, -1]
        test_X, test_y= test_upsampled[:, :-1], test_upsampled[:, -1]        
        model2 = deepcopy(model)
        model2.fit(train_X, train_y)
        
        # store metrics
        scores.append(model2.score(test_X, test_y))
        all_ypred.extend(model2.predict(test_X))
        all_ytrue.extend(test_y)
        
    return scores, all_ytrue, all_ypred