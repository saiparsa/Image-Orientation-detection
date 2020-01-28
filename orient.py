# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 16:55:53 2019

@author: saipr
"""
import pandas as pd
import numpy as np
import os
import sys
import pickle


    
    

#dataset function
def dataset(data):
  n=len(data)
  name=[]
  i=0
  while i<len(data):
    name.append(data[i])
    i=i+194
  orientation=[]
  j=1
  while j<len(data):
    orientation.append(int(data[j]))
    j=j+194
  img_data=[]
  k=2
  while k<n:
    img_data.append(data[k:k+192])
    
    k=k+194

  for a in range(len(img_data)):
      
    for k in range(len(img_data[a])):
        
      img_data[a][k]=int(img_data[a][k])
      
  img_data =np.asarray(img_data)
  
  orientation=np.asarray(orientation)
  
  return name,orientation,img_data

if __name__ == "__main__":
    #print('in main')
    if(len(sys.argv) != 5):
        raise(Exception("Error: expected 5 arguments"))
    mode=sys.argv[1]
    network=sys.argv[4]
    model_file=sys.argv[3]
    #print(mode)
    f=open(sys.argv[2],"r")
    train=f.read().split()
        #train data
    train_name,train_y,x_train=dataset(train)
    if (mode=="train" and network=="tree"):
        
        #train dec tree
        train_y=train_y.reshape(x_train.shape[0],1)
        data = np.append(x_train, train_y, 1)
        #test data
        
        def labels(Data):
            
            count = {} 
            for row in Data:
                label = row[-1]
                if label not in count:
                    count[label] = 0
                count[label] += 1
            return count
        
        class Condition:
            
            def __init__(self, feature, value):
                self.feature = feature
                self.value = value
        
            def match(self, example):
                return example[self.feature] >= self.value
        
        def split(Data, condition):
            
            true_rows, false_rows = [], []
            for row in Data:
                if condition.match(row):
                    true_rows.append(row)
                else:
                    false_rows.append(row)
            return true_rows, false_rows
        
        def gini_impurity(Data):
            
            counts = labels(Data)
            impurity = 1
            for label in counts:
                prob= counts[label] / float(len(Data))
                impurity -= prob**2
            return impurity
        
        def information_gain(left, right, uncertainty):
            
            p = float(len(left)) / (len(left) + len(right))
            return uncertainty - p * gini_impurity(left) - (1 - p) * gini_impurity(right)
        
        def best_split(Data):
            best_gain = 0  
            best_condition = None 
            uncertainty = gini_impurity(Data)
            for col in range(len(Data[0]) - 1):  
                unique_values = set([row[col] for row in Data])  
                # print(unique_values)
                #for val in range(len(unique_values)):
                val= len(unique_values)-1
                while val >0:  
                    unique_values=list(unique_values)
                    condition = Condition(col, unique_values[val])
        
                    true_rows, false_rows = split(Data, condition)
                    
                    if len(true_rows) == 0 or len(false_rows) == 0:
                        continue
        
                    gain = information_gain(true_rows, false_rows, uncertainty)
        
                    if gain > best_gain:
                        best_gain, best_condition = gain, condition
                    
                    val=val-20
                    # if val>192:
                    #     val=192
            
            return best_gain, best_condition
        
        class Leaf:
            
            def __init__(self, rows):
                self.predictions = labels(rows)
        
        class Decision_Node:
            
            def __init__(self, condition, right_branch, left_branch):
                self.condition = condition
                self.right_branch = right_branch
                self.left_branch = left_branch
        
        def build_tree(rows,depth,maxdepth):
            
        
            gain, condition = best_split(rows)
        
            #print (depth)
            if depth>maxdepth:
                return Leaf(rows)
        
           
            true_rows, false_rows = split(rows, condition)
        
            
            right_branch = build_tree(true_rows, depth+1, maxdepth)
        
            left_branch = build_tree(false_rows, depth+1, maxdepth)
        
            return Decision_Node(condition, right_branch, left_branch)
        
        my_tree = build_tree(data,0,3)
        file=open(model_file,'wb')
        
        pickle.dump(my_tree,file)
        
    if (mode=="train" and (network=="nnet" or network=="best")):        
        f=open("train-data.txt","r")
        train=f.read().split()
        train_name,train_y,x_train=dataset(train)
        train_y=np.asarray(train_y)
        y_train=[]
        for i in train_y:
            if i==0:
                y_train.append([1,0,0,0])
            if i==90:
                y_train.append([0,1,0,0])
            if i==180:
                y_train.append([0,0,1,0])
            if i==270:
                y_train.append([0,0,0,1])

        y_train=np.asarray(y_train)
        
        #print(train_y)
        #print(x_train)
        
        import numpy as np
        import math
        # import keras
        # from sklearn.metrics import accuracy_score as score
        # from sklearn.model_selection import KFold
        
        masks=[]
        class NeuralNetwork(object):
          def __init__(self,epochs, learning_rate):
            self.epochs=epochs
            self.learning_rate = learning_rate
            pass
          
          def fit(self,X1,y1):
            
            grads = {}                           
                                       
            n_x=192
            n_h=128
            n_y=4 
            
            accuracy=[]
            param=[]
               
            parameters = self.initialize(n_x, n_h, n_y)
            # k_fold = KFold(n_splits=5)#5 fold cross validation
            # for train_index, test_index in k_fold.split(X1):
            #   #print(test_index,train_index)
            #   X=X1[train_index]
            #   x_val=X1[test_index]
            #   Y=y1[train_index]
            #   y_val= y1[test_index]
        
            
            minibatches = self.random_mini_batches(X1, y1, X1.shape[0]%5)
            for i in range(0,self.epochs):
               # print(i)
                for batch in minibatches:
        
                    W1 = parameters["W1"]
                    b1 = parameters["b1"]
                    W2 = parameters["W2"]
                    b2 = parameters["b2"]
                    W3 = parameters["W3"]
                    b3 = parameters["b3"]
                        
                    X,Y = batch
                    #print(b1.shape,b2.shape,b3.shape)
                    m = X.shape[0]
                    A1, cache1 = self.linear_activation_forward(X, W1, b1, 'relu',True)
                    A2, cache2 = self.linear_activation_forward(A1, W2, b2, 'relu',True)
                    A3, cache3 = self.linear_activation_forward(A2, W3, b3, 'softmax',True)
                    
                    #print(cache3[0][0].shape,cache3[0][1].shape,cache3[0][2].shape)
                    #print(cache3[1].shape)
                    cost,A3 = self.compute_cost(A3, Y)
                    
                    #print(cost)
                    #dA3 = - (np.divide(Y, A3))
                    dA3= A3-Y 
                    #print(np.isnan())
                    #print(dA3)
                    
                    dA2, dW3, db3 = self.linear_activation_backward(dA3, cache3, 'softmax',True)
                    dA1, dW2, db2 = self.linear_activation_backward(dA2, cache2, 'relu',True)
                    dA0, dW1, db1 = self.linear_activation_backward(dA1, cache1,'relu',True)
                    #print(np.isnan(db2))
                    grads['dW1'] = dW1
                    grads['db1'] = db1
                    grads['dW2'] = dW2
                    grads['db2'] = db2
                    grads['dW3'] = dW3
                    grads['db3'] = db3
                    #print(dW1.shape,db1.shape,dW2.shape,db2.shape,dW3.shape,db3.shape)
        
                    parameters = self.update_parameters(parameters, grads, self.learning_rate)
                    
                    W1 = parameters["W1"]
                    b1 = parameters["b1"]
                    W2 = parameters["W2"]
                    b2 = parameters["b2"]
                    W3 = parameters["W3"]
                    b3 = parameters["b3"]
                    #print(W1.shape,b1.shape,W2.shape,b2.shape,W3.shape,b3.shape)
                # accuracy.append(self.evaluate(x_val,y_val,parameters))
                # print("Accuracy for fold",len(accuracy),":",accuracy[-1])
                # param.append(parameters)
        
            #best=np.argmax(accuracy)
            return parameters
            pass
          def initialize(self,n_x, n_h, n_y):
          
            W1 = np.random.randn(n_h, n_x)*0.01
            b1 = np.zeros(shape=(1,n_h))
            W2 = np.random.randn(n_h, n_h)*.01
            b2 = np.zeros(shape=(1,n_h))
            W3 = np.random.randn(n_y, n_h)*.01 
            b3 = np.zeros(shape=(1,n_y))
        
            parameters = {"W1": W1,"b1": b1,"W2": W2,"b2": b2,"W3": W3,"b3": b3}
            
            return parameters
          
          def linear_activation_forward(self,A_prev, W, b, activation,dropout):
            
            if activation == "softmax":
                
                Z, linear_cache = self.linear_forward(A_prev, W, b)
                #print(linear_cache[2].shape)
                A, activation_cache = self.softmax(Z)
                
            elif activation == "relu":
                
                Z, linear_cache = self.linear_forward(A_prev, W, b)
                #print(linear_cache[2].shape)
                A, activation_cache = self.relu(Z)
                if dropout:
                  mask = np.random.binomial(1, 0.8, size=A.shape)#dropout: drop_prob=0.2,keep_prob=0.8
                  masks.append(mask)
                  A=A*mask
               
            cache = (linear_cache, activation_cache)
            
            return A, cache
          
          def linear_forward(self,A, W, b):
            Z = np.dot(A,W.T ) + b
            cache = (A, W, b)
            return Z, cache
          
          def softmax(self,z):
        
            assert len(z.shape)==2
            s = np.max(z,axis=1)
            s=s[:,np.newaxis]
            e=np.exp(z-s)
            div=np.sum(e,axis=1)
            div=div[:,np.newaxis]
            
            return np.asarray(e/div),z
        
        
          def relu(self,z):
            out=np.maximum(0,z)
            
            return out,z
        
          def compute_cost(self,A3, Y):
            
            
            A3_d=[]
            '''for row in range(0,A3.shape[0]):
              max_ind=np.argmax(A3[row][:])
              A3_d.append(max_ind)
            
        
            AL = keras.utils.to_categorical(A3_d, 10)
            '''
            m = Y.shape[1]
            i =Y.shape[0]
            
            
            cost=0
            x=0
            for n in range(0,i):
              for k in range(0,m):
                x=Y[n][k]*np.log(A3[n][k])
                cost=cost+x
            #cost=keras.losses.sparse_categorical_crossentropy(AL,Y)
            return -cost/m,A3
            
            #return cost,AL.T
        
          def linear_activation_backward(self,dA, cache, activation,dropout):
            
            linear_cache, activation_cache = cache
            #print(activation_cache)
            
            if activation == "relu":
                
                dZ = self.relu_backward(dA, activation_cache)
                if dropout:
                  mask=masks.pop()
                  dZ= dZ*mask
            elif activation == "softmax":
                dZ = self.softmax_backward(dA, activation_cache)
            
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            
            return dA_prev, dW, db
        
          def relu_backward(self,dout,cache):
            dx,x=None,cache
            for i in range(x.shape[0]):
              for j in range(x.shape[1]):
                if x[i][j]>0:
                  x[i][j]=1
                else:
                  x[i][j]=0
        
            dx=np.multiply(x,dout)
            #print (dx)
            return dx
          
          def softmax_backward(self,dout, cache):
            out1=1
            S=self.softmax(cache)[0]
            '''print(cache)
            for i in range(0,S.shape[0]):
              
                if(S[i].all()==cache[i].all()): 
                  out=np.multiply(S,(1-S))
                else:
                  out=np.multiply(S,(0-S))
                #print(out)
                out1=np.multiply(out1,out)
            #print('out',out1)
            z=np.multiply(cache,out1)
            #print('sd',z)'''
            z=np.multiply(S*(1-S),dout)
            return z
          def linear_backward(self,dZ, cache):
            
            A_prev, W, b = cache
            
            m = A_prev.shape[1]
        
            
            dW = np.dot(dZ.T,cache[0]) / m
            #print('dz',dZ)
            db = np.squeeze(np.sum(dZ.T, axis=1, keepdims=True)) / m
            
            #print('db',db)
            dA_prev = np.dot(dZ,cache[1])
            
            np.asarray(db)
            #print(dw.shape,d.shape,dA_prev.shape)    
            return dA_prev, dW, db
        
          def update_parameters(self,parameters, grads, learning_rate):
            
            L = len(parameters) // 2 
            beta=0.9
            
            sdw=0
            sdb=0
            v={}
            for l in range(L):
                v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l+1)])
                v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l+1)])
                
                v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads['dW' + str(l + 1)]
                v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads['db' + str(l + 1)]
                parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]
                parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]
            
            return parameters
          def random_mini_batches(self,X, Y, mini_batch_size):
            X=X.T
            Y=Y.T
            m = X.shape[1]                  
            mini_batches = []
                
            
            permutation = list(np.random.permutation(m))
            shuffled_X = X[:, permutation]
            shuffled_Y = Y[:, permutation].reshape((4,m))
        
            
            num_complete_minibatches = math.floor(m/mini_batch_size) 
            for k in range(0, num_complete_minibatches):
                
                mini_batch_X = shuffled_X[:,k * mini_batch_size:(k + 1) * mini_batch_size]
                mini_batch_Y = shuffled_Y[:,k * mini_batch_size:(k + 1) * mini_batch_size]
              
                mini_batch = (mini_batch_X.T, mini_batch_Y.T)
                mini_batches.append(mini_batch)
            
            
            if m % mini_batch_size != 0:
                end = m - mini_batch_size * math.floor(m / mini_batch_size)
                mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size:]
                mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size:]
                mini_batch = (mini_batch_X.T, mini_batch_Y.T)
                mini_batches.append(mini_batch)
            
            return mini_batches
        
          def dropout(self,X, drop_probability):
            keep_probability = 1 - drop_probability
            mask = np.random.binomial(1, keep_probability, size=X.shape)
           
            X=X*mask
        
            return X
          
        out=NeuralNetwork(100,.01)
        
        p=out.fit(x_train,y_train)
        file=open(model_file,'wb')
        
        pickle.dump(p,file)
        
    if(mode=="test" and network=="tree"):
        f1=open(sys.argv[2],"r")
        test=f1.read().split()
        test_name,test_y,x_test=dataset(test)
        test_y=test_y.reshape(x_test.shape[0],1)
        test = np.append(x_test, test_y, 1)
        #test dec tree data
        #print(test)
        def labels(Data):
            
            count = {} 
            for row in Data:
                label = row[-1]
                if label not in count:
                    count[label] = 0
                count[label] += 1
            return count
        
        class Condition:
            
            def __init__(self, feature, value):
                self.feature = feature
                self.value = value
        
            def match(self, example):
                return example[self.feature] >= self.value
        
       
        class Leaf:
            
            def __init__(self, rows):
                self.predictions = labels(rows)
        
        class Decision_Node:
            
            def __init__(self, condition, right_branch, left_branch):
                self.condition = condition
                self.right_branch = right_branch
                self.left_branch = left_branch
        
        
        def classify(row, node):
            
            if isinstance(node, Leaf):
                return node.predictions
        
            if node.condition.match(row):
                return classify(row, node.right_branch)
            else:
                return classify(row, node.left_branch)
        
        
        correct =0
        dec_tree_preds=[]
        file = open('model_file.txt', 'rb')

        # dump information to that file
        my_tree = pickle.load(file)
        
        # close the file
        file.close()
        #preds=[]
        for row in test:
            out=(classify(row, my_tree))
            max_key = max(out, key=lambda k: out[k])
            dec_tree_preds.append(max_key)
            if (row[-1]==max_key):
                correct=correct+1
        dec_tree_acc=correct/test.shape[0]
        
        print ("Accuracy is", dec_tree_acc*100)
        outs=[]
        for i in range(len(test_name)):
            form=str(test_name[i]+" "+str(dec_tree_preds[i])+"\n")
            outs.append(form)
        file=open("output.txt","w")
        for i in range(len(outs)):
            j = file.write(outs[i])
        file.close()
        
    if(mode=="test" and (network=="nnet" or network== "best")):    
        f1=open("test-data.txt","r")
        test=f1.read().split()
        test_name,test_y,x_test=dataset(test)
        
        num_classes = 4
        test_y=np.asarray(test_y)
        y_test=[]
        for i in test_y:
            if i==0:
                y_test.append([1,0,0,0])
            if i==90:
                y_test.append([0,1,0,0])
            if i==180:
                y_test.append([0,0,1,0])
            if i==270:
                y_test.append([0,0,0,1])
        
        y_test=np.asarray(y_test)
        
        
        def linear_activation_forward(A_prev, W, b, activation,dropout):
            
            if activation == "softmax":
                
                Z, linear_cache = linear_forward(A_prev, W, b)
                #print(linear_cache[2].shape)
                A, activation_cache = softmax(Z)
                
            elif activation == "relu":
                
                Z, linear_cache = linear_forward(A_prev, W, b)
                #print(linear_cache[2].shape)
                A, activation_cache = relu(Z)
                if dropout:
                  mask = np.random.binomial(1, 0.8, size=A.shape)#dropout: drop_prob=0.2,keep_prob=0.8
                  masks.append(mask)
                  A=A*mask
               
            cache = (linear_cache, activation_cache)
            
            return A, cache
          
        def linear_forward(A, W, b):
            
            Z = np.dot(A,W.T ) + b
            cache = (A, W, b)
            return Z, cache
          
        def softmax(z):
        
            assert len(z.shape)==2
            s = np.max(z,axis=1)
            s=s[:,np.newaxis]
            e=np.exp(z-s)
            div=np.sum(e,axis=1)
            div=div[:,np.newaxis]
            
            return np.asarray(e/div),z
        
        
        def relu(z):
            out=np.maximum(0,z)
            
            return out,z
            
         
        
        def evaluate(x,y,parameters):
           
            #print(y)
            W1=parameters['W1']
            b1=parameters['b1']
            W2=parameters['W2']
            b2=parameters['b2']
            W3=parameters['W3']
            b3=parameters['b3']
        
            A1, cache1 = linear_activation_forward(x, W1, b1, 'relu',False)
            A2, cache2 = linear_activation_forward(A1, W2, b2, 'relu',False)
            A3, cache3 = linear_activation_forward(A2, W3, b3, 'softmax',False)
            
            #print(sum(A3[0]))
            #print(A3)
            #predict=np.argmax(y,axis=1)
            A3_d=[]
            predictions=[]
            for row in range(0,A3.shape[0]):
              max_ind=np.argmax(A3[row][:])
              A3_d.append(max_ind)
              #print(A3_d)
              predictions.append(max_ind*90)
            predicted=[]
            for i in A3_d:
                if i==0:
                    predicted.append([1,0,0,0])
                if i==1:
                    predicted.append([0,1,0,0])
                if i==2:
                    predicted.append([0,0,1,0])
                if i==3:
                    predicted.append([0,0,0,1])
        
            predicted=np.asarray(predicted)
            #print(predicted)
            #print(y.shape)
            #predicted = keras.utils.to_categorical(A3_d, 4)
            
            #predicted = keras.utils.to_categorical(predict, 10)
            #print(predicted.shape)
            wrong_pred=0
            for i in range(0, predicted.shape[0]):
              for j in range(0,predicted.shape[1]):
                if (predicted[i][j] != y[i][j]):
                  wrong_pred=wrong_pred+1
                  break
            accuracy = ((predicted.shape[0]-wrong_pred)/predicted.shape[0])
            #print(predicted)
            #accuracy=score(predicted,y)
            
            #print(accuracy*100)
            return accuracy*100,predictions
            pass
        
        file = open('model_file.txt', 'rb')

        # dump information to that file
        p = pickle.load(file)
        
        # close the file
        file.close()
        
        nnet_accuracy,pred=evaluate(x_test,y_test,p)
        outs=[]
        for i in range(len(test_name)):
            form=str(test_name[i]+" "+str(pred[i])+"\n")
            outs.append(form)
        file=open("output.txt","w")
        for i in range(len(outs)):
            j = file.write(outs[i])
        file.close()
        
        print("accuracy is", nnet_accuracy)
        
    if (mode=="train" and network=="nearest"):
        print('train')
        f=open(sys.argv[2],"r",encoding='utf-8')
        train=f.read()
        file=open(model_file,'w')
        for i in range(len(train)):
            j = file.write(train[i])
        file.close()
        
        
    if (mode=="test" and network=="nearest"): 
        
        file = open('model_file.txt', 'r',encoding='utf-8')
        train1 = file.read().split()
        train_name,train_y,train_x=dataset(train1)
        #print(len(train_x))
        f1=open(sys.argv[2],"r")
        test=f1.read().split()
        test_name,test_y,test_x=dataset(test)
        X_train=np.asarray(train_x)
        #print(len(test_x))
        
        def KNN(inp,x_trn,y_trn,k):
          euclid_dist={}
          for i in range(len(x_trn)):
              euclid_dist[i]=(np.sqrt(np.sum(np.square(inp-x_trn[i,:]))))
          p=sorted(euclid_dist.items(), key =lambda dic:(dic[1], dic[0]))
          preds={}
          for i in range(k):
            if y_trn[p[i][0]] in preds.keys():
              preds[y_trn[p[i][0]]]=preds[y_trn[p[i][0]]]+1
            else:
              preds[y_trn[p[i][0]]]=1
          q=sorted(preds.items(), key =lambda dic:(dic[1], dic[0]))
          prediction=q[-1][0]
          return prediction
        
        knn_preds=[]
        n=len(test_y)
        for i in range(n):
          knn_preds.append(KNN(test_x[i],X_train,train_y,5))
        
        count=0
        for i in range(n):
          if knn_preds[i]==test_y[i]:
            count=count+1
        
        knn_accuracy=count/n
        outs=[]
        for i in range(n):
            form=str(test_name[i]+" "+str(knn_preds[i])+"\n")
            outs.append(form)
        file=open("output.txt","w")
        for i in range(len(outs)):
            j = file.write(outs[i])
        file.close()
        print("Accuracy is:",knn_accuracy*100)
       

