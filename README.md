# aknagu-saedup-saiparsa-a4


a4 created for aknagu-saedup-saiparsa

 Tree:- (Decion tree)

We have used classification and regression trees (CART ) algorithm for implementing decison tree in our program.

We have defined  the functions for perfroming various operations in the algorithm like 
counting the number of times each example occuring in the dataset,
GINI impurity (we have used GINI impurity as a cost metric to evalutate splits in the dataset,
and information gain of the split in the  dataset. 

We are finding the best split for the data by iterating over the dataset and considering split after every feature.
The split which gives the maximum information gain will be taken as the best split for the data taken.

Whenever a split is made ,data is considered into two branches considering a decison question ,
data satisying that question will be taken into one branch and not satisying that question will be taken into another branch.

We have recursively split each branch of our  into further branches till we get information gain for the split 

or till the maximum depth defined by us is reached. (we have taken the maximum depth to be 3 for our decison tree). 

The data present in the terminal  branch is considered to be  classified as a label to that branch.

The decison tree formed in the above process from the training data is used on the test data and the label of the terminal branches is 
said to be the prediction with the testing data. 

As we increase the depth of the decison tree the model is getting over fitted and giving less accurcy on the test data. 

The running time of the model is increasing exponentially as we increase the depth of the tree.

Nearest (KNN ): -

In this algorithm we are considering the Euclidean distances to all the training examples from a particular test image and inserted them into an ordered(ascending) dictionary and from this we considered top 5 closest distances.

From these selected 5 least distances we check which labels are generating such distances and the ones that are contributing the maximum to these 5 selected distances(label which is repeated the most) is considered as our prediction.

If there exists a tie in the contributed labels we consider the label which results the least distance as the prediction.

The process is repeated for the entire test data and the labels are predicted for all the test cases.

Pros of KNN:
No training i.e, creating a model and fiddeling with parameters is required.

Cons:
Very time consuming if the training data is too large to deal with.

We got the best accuracy for 5 nearest neighbours of 70.14%
We tried the same algorithm for different k values:
   
   K values      Accuracy
      2            66.4
      3            68.7
      4            68.6
      5            70.1
      6            68.6
      7            68.7
      8            69.1
      9            69.4
   

We could obsever any particukar trend in relationship between the k value and acccuracy, we couldn't try higher values as they were computationally expensive 

Neural Network:

We used a fully connected feed forward neural network with 2 hidden layers. The input layer has 192 neurons inline with the number of 
features and each of the hidden layers have 128 neurons with relu activation function.
We have considered sigmoid and relu as our activation function thoufh sigmoid was giving slightly more accuracy it was not consistent 
with the accuracy once we started changing few hyperparameters. We considered relu and mathematically represented it and its derivative 
and used then in Forward, Back Propagation respectively.
As our output is multi class classification with 4 classes we choose 4 neurons in th eoutput layer with softmax as activation function.
For all the derivatives and mathematical representations we coded everything from scrath and not used any other libraries other than numpy.

Data:
For this task we converted the orientations into thier one hot encoded form.

After considering a basic network, we tried implementing the code on a small subset of the data.We observed that the training data is not skewed and had all the output classed examples in equal proportion. We then modified our model and arrived at the above mentioned architecture.

Random Mini batches:
We tried reducing the training time,we implemented the random minibatches in which we spilt the training data into batches anf trained the model on that, we have written the code for the random mini batch from scratch and used only numpy library.We tried different batch sizes and it wasn't effecting anything except running time.

K fold:
We tried to build the model as robust as possible, we wanted to reduce the effect of overfitting hence we implemented k fold cross validation and inmplemnted 5 fold CV we implemented this using sk learn and got favourable results on the test data.

Dropout:
As we couldn't replicate the k fold and implement it from scratch we tried to insert dropout of 20% which gave us equally interesting results as the kfold.

Gradient Descent:
We used teh learning rate of 0.01 after considering various other learning rates, we observed a good decrease in the cost for every epoch.But we tried to increase its effectiveness.

Rms Prop:
Implemented RMS prop over the gradient descent and got better results than the gradient descent algorithm.

Our model gave the best accuracy of 76% for 100 epochs.


Ref: Deeplearning.ai Deep Learning tutorials
     Decision Tree:https://www.youtube.com/watch?v=LDRbO9a6XPU&feature=youtu.be


PS : Please rename the uploaded model files to model_file while running the test on them.
--
