import os
import jieba
import jieba.posseg as pseg
import sys
import string
import shutil
import csv
import numpy as np
from collections import defaultdict
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize,word_tokenize



def travel(rootDir): 
    for lists in os.listdir(rootDir): 
        path = os.path.join(rootDir, lists) 
        if os.path.isfile(path):
            statinfo = os.stat(path)
            if statinfo.st_size ==0: print path
#            print statinfo.st_size
            path = path.replace("segfile\\","")
            filelist.append(path)

        if os.path.isdir(path): 
            travel(path) 

def fenci(path) :
#    source_path = "files"
    sFilePath = 'segfile'
    if not os.path.exists(sFilePath) : 
        os.mkdir(sFilePath)

    f = open(source_path+"\\"+path,'r+')
    file_list = f.read()
    f.close()

    seg_list = jieba.cut(file_list,cut_all=True)


    result = []
    for seg in seg_list :
      seg = ''.join(seg.split())
      if (seg != '' ) :
            result.append(seg)

    f = open(sFilePath + "/" + path,"w+")
    
    f.write(' '.join(result))
    f.close()


        
if __name__=="__main__":
#    nltk.download()
#    reload(sys)
#    sys.setdefaultencoding('utf-8')

    vocab_size = 150
    source_path = "segfile"
    filelist = []
    travel(source_path)
#    ps=PorterStemmer()
    
    
# =============================================================================
#     example_words=['python','pythoner','pythoning','pythoned','pythonly']
#     example_text="Remote Evaluation (AST evaluation), DW (3/23/01 10:03:32 AM)	Our deployment of class files for HCR and Evaluation does not support	remote targets.DW (4/25/01 2:03:16 PM)	Could provide a pluggable &quot;deployer&quot; onto a java debug target to handle.	We only provide the local implementation.DW (7/31/01 3:33:59 PM)	The JDK 1.4 spec uses the approach of replacing types over the wire.	A map of (old) reference types and a collection of bytes that should be loaded	as the new class are provided at the JDI layer. This takes care of the remote	case (i.e. JDI handles it).	This seems like a good design to follow. The part missing from the Sun 1.4	spec is a way to distinguish versions of types.DW (8/3/01 9:36:14 AM)	In the case of evaluataion we need to deploy new class files. In the case of	HCR we need to deploy if remote otherwise deployment is already done by	the builder.	The launcher knows how to deploy class files - it knows where the target is	and the classpath of the target. Thus it should supply the deployment policy.	The new JDI API for HotSwap takes care of loading (but not deployment) - i.e. bytes over the wire.	For evaluation we need a deployment policy since the types are new.	IDeploymentPolicy - public void deploy(byte[][] classFileBytes String[][] typeNames) throws DebugException.	The IJavaDebugTarget will support a #setDeploymentPolicy(IDeploymentPolicy) which the	launcher will provide.	For hotswap we could still deploy files and then also use JDI API's to notify target	of updates.DW (9/17/01 11:32:05 AM)	Using AST evaluation would not require file deployment"
# 
# 
#     list_sentences=sent_tokenize(example_text)
# 
#     list_words=word_tokenize(example_text)
#     
#     list_stemWords=[ps.stem(w) for w in example_words]
#     
#     
#     list_stemWords1=[ps.stem(w) for w in list_words]
# =============================================================================
    
    
# =============================================================================
#     print " begin jieba"    
#     
#     
#     tobecheckdir = "segfile"
#     if os.path.isdir(tobecheckdir):
#        shutil.rmtree('segfile')  
#     shutil.copytree(source_path,"segfile")
#     seg_path = "segfile/"
#     file_seg_list= []
#     for ff in filelist :
#         c_path = ff
#         sFilePath = 'segfile'
#         if not os.path.exists(sFilePath) : 
#             os.mkdir(sFilePath)
#     
#         
#         file_single = []
#         fin = open(source_path+"\\"+c_path, 'r+')
#         for eachLine in fin:  
#            line = eachLine.strip().decode('utf-8', 'ignore') 
#            seg_line = jieba.cut(line,cut_all=True)
#            result = []
#            for seg in seg_line:
#                seg = ''.join(seg.split())
#                if (seg != '' ):
#                   result.append(seg)
#            if len(result)>0:
#                file_single.append(result)
#         fin.close()
# 
#         print "testing"
#         print sFilePath + "/" + c_path
#         f = open(sFilePath + "/" + c_path,"w+")
# #        print file_single
#         for file_line in file_single:
#            f.write(' '.join(file_line))
#            f.write("\n")
#         f.close()
#            
#     
#         seg_list = jieba.cut(filelist,cut_all=True)
#     
#     
#         result = []
#         for seg in seg_list :
#           seg = ''.join(seg.split())
#           if (seg != '' ) :
#                 result.append(seg)
#     
#         f = open(sFilePath + "/" + c_path,"w+")
#         
#         f.write(' '.join(result))
#         print result
#         f.close()
# =============================================================================
                
    corpus = [] 
    for ff in filelist :
        fname = source_path + "/" + ff
        f = open(fname,'r+')
        content = f.read()
        f.close()
        corpus.append(content) 
    
    tokenize = lambda doc: doc.lower().split(" ")
    print 'start skl_tfidf...'
    MAX_FEATURE = 300
    try:
        sklearn_tfidf = TfidfVectorizer(norm= None, min_df=0, use_idf=True, smooth_idf=True, max_features=MAX_FEATURE,
                                    sublinear_tf=True, tokenizer=  tokenize, decode_error='ignore' )
        skl_tfidf_representation = sklearn_tfidf.fit_transform(corpus).toarray()
    except MemoryError:
        print 'memory error'
   
    print " \n begin write tfidf files"  
    tobecheckdir = "tfidf"
# =============================================================================
#     if os.path.isdir(tobecheckdir):
#         shutil.rmtree('tfidf')  
#     shutil.copytree("segfile","tfidf")
# =============================================================================
    sFilePath = 'tfidf'
 
    for i in range(len(filelist)) :
         path = sFilePath + "/"+filelist[i]
         print path
         f = open(path,'w+')
         for j in range(MAX_FEATURE) :
             if j==0: 
                 f.write(str(skl_tfidf_representation[i][j]))
             else:
                 f.write(", " + str(skl_tfidf_representation[i][j]))
 #            f.write(word_sort[j]+"    "+str(weight_sort[i][j])+"\n")
         f.write("\n")
         f.close()    


    

    
        


         
        

