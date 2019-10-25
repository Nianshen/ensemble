from __future__ import division
import xml.etree.ElementTree as ET
import random
import numpy as np
import math
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mutual_info_score
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
np.set_printoptions(threshold=np.inf)  
import os


def XML2arrayRAW(neg_path, pos_path):
    reviews = []
    negReviews = []
    posReviews = []

    neg_tree = ET.parse(neg_path)
    neg_root = neg_tree.getroot()
    for rev in neg_root.iter('review'):
        reviews.append(rev.text)
        negReviews.append(rev.text)



    pos_tree = ET.parse(pos_path)
    pos_root = pos_tree.getroot()

    for rev in pos_root.iter('review'):
        reviews.append(rev.text)
        posReviews.append(rev.text)

    return reviews,negReviews,posReviews

def GetTopNMI(n,CountVectorizer,X,target):
    MI = []
    length = X.shape[1]


    for i in range(length):
        temp=mutual_info_score(X[:, i], target)
        MI.append(temp)
    MIs = sorted(range(len(MI)), key=lambda i: MI[i])[-n:]
    return MIs,MI


def getCounts(X,i):

    return (sum(X[:,i]))

def extract_and_split(neg_path, pos_path):
    reviews,n,p = XML2arrayRAW(neg_path, pos_path)
    #train, train_target, test, test_target = split_data_balanced(reviews,1000,200)
    train=reviews
    train_target=[]
    test = []
    test_target=[]
    train_target = [0]*1000+[1]*1000
    return train, train_target, test, test_target


def evaluate(predict_y, true_y):
  
    TP, TN, FP, FN = 0, 0, 0, 0
    len_num = len(true_y)
    for i in range(len_num):
        print i
        if int(true_y[i]) == 1 and int(predict_y[i]) == 1:
            TP += 1
        elif int(true_y[i]) == 1 and int(predict_y[i]) == 0:
            FN += 1
        elif int(true_y[i]) == 0 and int(predict_y[i]) == 1:
            FP += 1
        else:
            TN += 1
    fz=TP+TN
    print fz
    fm=TP +TN + FP+FN
    print fm
    accuracy = fz/fm

    print("----------------------------------------")
    print("         | Predict good | Predict bad   ")
    print("----------------------------------------")
    print("True good|     {0}      |    {1}        ".format(TP, FN))
    print("----------------------------------------")
    print("True bad |     {0}      |    {1}        ".format(FP, TN))
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print accuracy
    recall = TP / (TP + FN)
    f_measure = (2 * TP) / (2 * TP + FP + FN)
    g_mean = math.sqrt((TP / (TP + FN)) * (TN / (TN + FP)))
    print('accuracy=', accuracy, ',recall=', recall, ',f_measure=', f_measure, ',g_mean=', g_mean)
#    #print "accuracy="+ str(accuracy)
    return accuracy
#print len(domain)
#opt.annealingoptimize(domain,costf,T=2000.0,cool=0.95,step=1):

#print configcost(s)
#print s

def sent(src,src_k,src_d,dest,pivot_num,pivot_min_st,dim,c_parm):
    pivotsCounts = []
    pivotsCounts_k = []
    pivotsCounts_d = []
    
    
    #get representation matrix

    weight_str = src + "_to_" + dest + "/weights/w_" + src + "_" + dest + "_" + str(pivot_num) + "_" + str(
        pivot_min_st) + "_" + str(dim)+".npy"
    weight_str_k = src_k  + "_to_" + dest + "/weights/w_" + src_k  + "_" + dest + "_" + str(pivot_num) + "_" + str(
        pivot_min_st) + "_" + str(dim)+".npy"
    weight_str_d = src_d  + "_to_" + dest + "/weights/w_" + src_d  + "_" + dest + "_" + str(pivot_num) + "_" + str(
        pivot_min_st) + "_" + str(dim)+".npy"
    #gets the model weights 
    mat= np.load(weight_str)
    mat_k= np.load(weight_str_k)
    mat_d= np.load(weight_str_d)
    
    #gets the encoder as the transformation function
    mat = mat[0]
   # print mat.shape
    mat_k = mat_k[0]
    mat_d = mat_d[0]
    
    #print mat.shape

    filename = src + "_to_" + dest + "/split/"
    if not os.path.exists(os.path.dirname(filename)):
        #gets all the train and test for sentiment classification
        train, train_target, test, test_target = extract_and_split("data/"+src+"/negative.parsed","data/"+src+"/positive.parsed")
    else:
        with open(src + "_to_" + dest + "/split/train", 'rb') as f:
            train = pickle.load(f)
        with open(src + "_to_" + dest + "/split/test", 'rb') as f:
            test = pickle.load(f)
        with open(src + "_to_" + dest + "/split/train_target", 'rb') as f:
            train_target = pickle.load(f)
        with open(src + "_to_" + dest + "/split/test_target", 'rb') as f:
            test_target = pickle.load(f)
            
            
            
    filename_k = src_k + "_to_" + dest + "/split/"
    if not os.path.exists(os.path.dirname(filename_k)):
        #gets all the train and test for sentiment classification
        train_k, train_target_k, test_k, test_target_k = extract_and_split("data/"+src_k+"/negative.parsed","data/"+src_k+"/positive.parsed")
    else:
        with open(src_k + "_to_" + dest + "/split/train", 'rb') as f1:
            train_k = pickle.load(f1)
        with open(src_k + "_to_" + dest + "/split/test", 'rb') as f1:
            test_k = pickle.load(f1)
        with open(src_k + "_to_" + dest + "/split/train_target", 'rb') as f1:
            train_target_k = pickle.load(f1)
        with open(src_k + "_to_" + dest + "/split/test_target", 'rb') as f1:
            test_target_k = pickle.load(f1)            


    filename_d = src_d + "_to_" + dest + "/split/"
    if not os.path.exists(os.path.dirname(filename_d)):
        #gets all the train and test for sentiment classification
        train_d, train_target_d, test_d, test_target_d = extract_and_split("data/"+src_d+"/negative.parsed","data/"+src_d+"/positive.parsed")
    else:
        with open(src_d + "_to_" + dest + "/split/train", 'rb') as f2:
            train_d = pickle.load(f2)
        with open(src_d + "_to_" + dest + "/split/test", 'rb') as f2:
            test_d = pickle.load(f2)
        with open(src_d + "_to_" + dest + "/split/train_target", 'rb') as f2:
            train_target_d = pickle.load(f2)
        with open(src_d + "_to_" + dest + "/split/test_target", 'rb') as f2:
            test_target_d = pickle.load(f2)            
            
            
    unlabeled, source, target = XML2arrayRAW("data/" + src + "/" + src + "UN.txt","data/" + dest + "/" + dest + "UN.txt")
    unlabeled_k, source_k, target_k = XML2arrayRAW("data/" + src_k + "/" + src_k + "UN.txt","data/" + dest + "/" + dest + "UN.txt")
    unlabeled_d, source_d, target_d = XML2arrayRAW("data/" + src_d + "/" + src_d + "UN.txt","data/" + dest + "/" + dest + "UN.txt")
#    #we add the train to the unlabeled list in order to get good vectorizer
    unlabeled = source + train+ target
    unlabeled_k = source_k + train_k+ target_k
    unlabeled_d = source_d + train_d+ target_d


    bigram_vectorizer_unlabeled = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=40, binary=True)
    bigram_vectorizer_unlabeled_k = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=40, binary=True)
    bigram_vectorizer_unlabeled_d = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=40, binary=True)
    X_2_train_unlabeled = bigram_vectorizer_unlabeled.fit_transform(unlabeled).toarray()
    X_2_train_unlabeled_k = bigram_vectorizer_unlabeled_k.fit_transform(unlabeled_k).toarray()
    X_2_train_unlabeled_d= bigram_vectorizer_unlabeled_d.fit_transform(unlabeled_d).toarray()

    filename = src + "_to_" + dest + "/" + "pivotsCounts/" + "pivotsCounts" + src + "_" + dest + "_" + str(
        pivot_num) + "_" + str(pivot_min_st)
    with open(filename, 'rb') as f:
        pivotsCounts = pickle.load(f)
    filename_k = src_k + "_to_" + dest + "/" + "pivotsCounts/" + "pivotsCounts" + src_k + "_" + dest + "_" + str(
        pivot_num) + "_" + str(pivot_min_st)
    with open(filename_k, 'rb') as fk:
        pivotsCounts_k = pickle.load(fk)    
    filename_d = src_d + "_to_" + dest + "/" + "pivotsCounts/" + "pivotsCounts" + src_d + "_" + dest + "_" + str(
        pivot_num) + "_" + str(pivot_min_st)
    with open(filename_d, 'rb') as fd:
        pivotsCounts_d = pickle.load(fd)    




    trainSent=train
    bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=20, binary=True)
    X_2_train = bigram_vectorizer.fit_transform(trainSent).toarray()
    X_2_test_unlabeld = bigram_vectorizer_unlabeled.transform(trainSent).toarray()
    #delete the pivots from the test matrix 
    XforREP = np.delete(X_2_test_unlabeld, pivotsCounts, 1)  # delete second column of C
   # print XforREP.shape
    rep = XforREP.dot(mat)
    print 'rep',rep.shape
    trainSent_k=train_k
    bigram_vectorizer_k = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=20, binary=True)
    X_2_train_k = bigram_vectorizer_k.fit_transform(trainSent_k).toarray()
    X_2_test_unlabeld_k = bigram_vectorizer_unlabeled_k.transform(trainSent_k).toarray()
    XforREP_k = np.delete(X_2_test_unlabeld_k, pivotsCounts_k, 1)  # delete second column of C
   # print XforREP_k.shape
    rep_k = XforREP_k.dot(mat_k)
    
    trainSent_d=train_d
    bigram_vectorizer_d = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=20, binary=True)
    X_2_train_d = bigram_vectorizer_d.fit_transform(trainSent_d).toarray()
    X_2_test_unlabeld_d = bigram_vectorizer_unlabeled_d.transform(trainSent_d).toarray()
    XforREP_d = np.delete(X_2_test_unlabeld_d, pivotsCounts_d, 1)  # delete second column of C
   # print XforREP_d.shape
    rep_d = XforREP_d.dot(mat_d)


    X_dev_test = bigram_vectorizer.transform(test).toarray()
    X_dev_test_unlabeled = bigram_vectorizer_unlabeled.transform(test).toarray()
    XforREP_dev = np.delete(X_dev_test_unlabeled, pivotsCounts, 1)  # delete second column of C
    XforREP_dev = XforREP_dev.dot(mat)
    devAllFeatures = np.concatenate((X_dev_test,XforREP_dev),1)
    X_dev_test_k = bigram_vectorizer_k.transform(test_k).toarray()
    X_dev_test_unlabeled_k = bigram_vectorizer_unlabeled_k.transform(test_k).toarray()
    XforREP_dev_k = np.delete(X_dev_test_unlabeled_k, pivotsCounts_k, 1)  # delete second column of C
    XforREP_dev_k = XforREP_dev_k.dot(mat_k)
    devAllFeatures_k = np.concatenate((X_dev_test_k,XforREP_dev_k),1)
    
    
    X_dev_test_d = bigram_vectorizer_d.transform(test_d).toarray()
    X_dev_test_unlabeled_d = bigram_vectorizer_unlabeled_d.transform(test_d).toarray()
    XforREP_dev_d = np.delete(X_dev_test_unlabeled_d, pivotsCounts_d, 1)  # delete second column of C
    XforREP_dev_d = XforREP_dev_d.dot(mat_d)
    devAllFeatures_d = np.concatenate((X_dev_test_d,XforREP_dev_d),1)



    allfeatures = np.concatenate((X_2_train, rep), axis=1)
    allfeatures_k = np.concatenate((X_2_train_k, rep_k), axis=1)
    allfeatures_d = np.concatenate((X_2_train_d, rep_d), axis=1)

    lbl_num = 1000
    dest_test, source, target = XML2arrayRAW("data/"+dest+"/negative.parsed","data/"+dest+"/positive.parsed")
    dest_test_target= [0]*lbl_num+[1]*lbl_num
    X_dest = bigram_vectorizer.transform(dest_test).toarray()
   # print 'xdest is',X_dest.shape
    X_2_test = bigram_vectorizer_unlabeled.transform(dest_test).toarray()
    XforREP_dest = np.delete(X_2_test, pivotsCounts, 1)  # delete second column of C
    rep_for_dest = XforREP_dest.dot(mat)
    allfeaturesFinal = np.concatenate((X_dest, rep_for_dest), axis=1)
   # print allfeaturesFinal.shape
    X_dest_k = bigram_vectorizer_k.transform(dest_test).toarray()
    X_2_test_k = bigram_vectorizer_unlabeled_k.transform(dest_test).toarray()
    XforREP_dest_k = np.delete(X_2_test_k, pivotsCounts_k, 1)  # delete second column of C
    rep_for_dest_k = XforREP_dest_k.dot(mat_k)
    allfeaturesFinal_k = np.concatenate((X_dest_k, rep_for_dest_k), axis=1)
    
    
    X_dest_d = bigram_vectorizer_d.transform(dest_test).toarray()
    X_2_test_d = bigram_vectorizer_unlabeled_d.transform(dest_test).toarray()
    XforREP_dest_d = np.delete(X_2_test_d, pivotsCounts_d, 1)  # delete second column of C
    rep_for_dest_d = XforREP_dest_d.dot(mat_d)
    allfeaturesFinal_d = np.concatenate((X_dest_d, rep_for_dest_d), axis=1)
    
    
#    reptest=np.concatenate((rep, rep_k,rep_d))
#    print reptest.shape

    logreg = LogisticRegression(C=c_parm)
    logreg.fit(X_2_train, train_target)
    lgs = logreg.score(X_dest, dest_test_target)
    log_dev_source = logreg.score(X_dev_test, test_target)
    clf1 =  LogisticRegression(C=c_parm)
    clf1.fit(allfeatures, train_target)
    pb= clf1.predict_proba( allfeaturesFinal)

    clf2 =  LogisticRegression(C=c_parm)
    clf2.fit(allfeatures_k, train_target_k)
##
    pk= clf2.predict_proba( allfeaturesFinal_k)
###    pc2=sum_cols(pk)
##    print pk[:10][:]
    clf3 =  LogisticRegression(C=c_parm)
    lf3.fit(allfeatures_d, train_target_d)

    pd= clf3.predict_proba(allfeaturesFinal_d)
    
    #######################################################
#     clf1 =  LogisticRegression(C=c_parm)
#     clf1.fit(X_2_train, train_target)
#     pb= clf1.predict_proba(X_dest)
#
# #    #print pb[:10][:]
# #
# ##    pc1=(sum_cols(pb))
# #    #print pb[:10][:]
#     clf2 =  LogisticRegression(C=c_parm)
#     clf2.fit(X_2_train_k, train_target_k)
# ##
#     pk= clf2.predict_proba( X_dest_k)
# ###    pc2=sum_cols(pk)
# ##    print pk[:10][:]
#     clf3 =  LogisticRegression(C=c_parm)
#     clf3.fit(X_2_train_d, train_target_d)
#
#     pd= clf3.predict_proba(X_dest_d)
    #####################################################################################
#    clf1 =  LogisticRegression(C=c_parm)
#    clf1.fit(rep, train_target)
#    pb= clf1.predict_proba(rep_for_dest)
#   # print rep_for_dest.shape
#    
##    #print pb[:10][:]
##    
###    pc1=(sum_cols(pb))
##    #print pb[:10][:]
#    clf2 =  LogisticRegression(C=c_parm)
#    clf2.fit(rep_k, train_target_k)
###    
#    pk= clf2.predict_proba(rep_for_dest_k)
####    pc2=sum_cols(pk)
###    print pk[:10][:]
#    clf3 =  LogisticRegression(C=c_parm)
#    clf3.fit(rep_d, train_target_d)
##
#    pd= clf3.predict_proba(rep_for_dest_d)
  #  label3=clf3.predict(X_dest_d)
   # print label3
    pred=0.1*pb+0.4*pk+0.5*pd
    labels=[]
    for i in range(2000):
        arr0=pred[i][0]
        arr1=pred[i][1]
    #print arr1
        if arr0>arr1:
            labels.append('0')
        else:
            labels.append('1')
    
    #print pred[:10]
   # predict_y = classifier.predict(test_X)
   # auc = evaluate_auc(predict_pro, test_y)
    acc=evaluate(labels, dest_test_target)
    print acc

