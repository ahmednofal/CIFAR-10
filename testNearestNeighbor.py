# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 19:22:30 2017

@author: mohamed
"""
import numpy as np
import pylab as pl
import NearestNeighbor as NN

def scatterplot2(scores,decision,marker1='o',marker2='^',label1='',label2='',transp=1.0):
    scores1 = scores[decision == 1]
    scores2 = scores[decision == 0]
    pl.scatter(scores1[:,0], scores1[:,1],edgecolors='face', marker=marker1, label=label1, c='g',alpha=transp)
    pl.scatter(scores2[:,0], scores2[:,1],edgecolors='face', marker=marker2, label=label2, c='r',alpha=transp)

# Use numpy to load the data contained in the file into a 2-D array
scores = np.loadtxt('../data/admission/scores.dat')
decision = np.loadtxt('../data/admission/decision.dat')
nn = NN.NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(scores, decision) # train the classifier on the training data and labels

#generate test points
scores_test = np.mgrid[10:71:1, 40:90:1].reshape(2,-1).T
decision_predict = nn.predict(scores_test,'L2') # predict labels on the test data

scatterplot2(scores,decision,'o','^','admitted','not admitted')
scatterplot2(scores_test,decision_predict,'o','^','','',0.35)
pl.grid()
pl.xlabel('exam1')
pl.ylabel('exam2')
pl.legend(loc='upper left')
pl.xlim(10., 70.)
pl.ylim(40., 90.)
pl.show()