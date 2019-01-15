import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import os
from scipy.special import comb, perm

def calculate_same_value(labels_sorted, test_p_sorted, start_pos):
    i = start_pos
    num_same = 0
    num_p = 0
    while test_p_sorted[start_pos]==test_p_sorted[i]:
        num_same = num_same + 1
        if labels_sorted[i]==1: num_p = num_p + 1
        i = i+1
        if i==len(labels_sorted)-1: break
    return num_p, num_same
        
    

def eval_y( test_p, test_y, labels):
#    labels: ground truth 0,1
#    test_y: predicted labels 0,1
#    test_p: predicted confidence numetric 
   

   test_p_sorted = test_p


   test_p_index = sorted(range(len(test_p_sorted)), key=lambda k: test_p_sorted[k], reverse=True)
   test_p_sorted = sorted(test_p, reverse=True)
 

   labels_sorted = []
   for index in test_p_index:
        labels_sorted.append(labels[index])
  

   
   top_num = 10
   top10rank = 0
   for i in xrange(top_num):
        if labels_sorted[i]==1:
            num_p, num_s = calculate_same_value(labels_sorted, test_p_sorted,i)
            num_r = top_num-i
#            print "num_same: ",num_s,"num_r: ", num_r, "num_p: ", num_p
            if num_p>(num_s-num_r): 
                top10rank = 1
                break
            v1 = perm(num_s-num_r, num_p)*perm(num_s-num_p, num_s-num_p)
            v2 = perm(num_s, num_s)
            top10rank = 1-(float)((float)(v1)/(float)(v2))
            if top10rank > 1: top10rank = 1
            if top10rank!=top10rank: top10rank = 1

            break 
   
   return  top10rank