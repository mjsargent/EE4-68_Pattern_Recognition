# mjsargent

#%%

# - This Python script should be run after the MATLAB script for generating 
# the required data splits. We chose to split the deep learning network and the     
# data handling between MATLAB and Python based on our coding proficencies and also
# the ease of data handling in MATLAB 
#
#
#Import required dependencies

import os # For file handling
import torch # PyTorch for neural net
from torch import nn
from torch.autograd import Variable
import scipy # For standard algorithms e.g kNN
import numpy as np # For array manipulation
import json # For loading features
from sklearn.neighbors import NearestNeighbors
from scipy.io import loadmat
os.chdir('/home/mjsargent/repos/PR/pattern-recognition/CW2/PR_data') # Change this directory to the directory you save the data to
with open('feature_data.json', 'r') as f:
    features = json.load(f)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Determine whether the computer has a dedicated GPU 

val_gal = np.array(loadmat('val_gallery.mat')['val_gallery']) # Features in validation gallery
val_que = np.array(loadmat('val_query.mat')['val_query'])     # Features in validation query

val_gal_labels = np.array(loadmat('val_gallery_label.mat')['val_gallery_label'])  # Validation labels
val_query_labels = np.array(loadmat('val_query_label.mat')['val_query_label'])

gallery_labels = np.array(loadmat('gallery_labels.mat')['gallery_labels']) # Test set labels
query_labels = np.array(loadmat('query_labels.mat')['query_labels']) # 

camId_v_gal = np.array(loadmat('val_gal_camId.mat')['val_gal_camId'])   # Camera IDs for validation and testing set
camId_v_quer = np.array(loadmat('val_quer_camId.mat')['val_quer_camId'])
camId_quer = np.array(loadmat('query_camId.mat')['query_camId'])
camId_gal = np.array(loadmat('gallery_camId.mat')['gallery_camId'])

class TripletLoss(nn.Module): # Triplet Loss implementation for anchor/positive/negative
                              # Sourced from: https://github.com/adambielski/siamese-triplet/blob/master/losses.py
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=False):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()
    

#%%


#%%


train_idxs = loadmat('train_idx.mat')['train_idx'].flatten()    # Load required lists of features + labels generated via MATLAB 
valid_idxs = loadmat('valid_idx.mat')['validation_idx'].flatten()
camID = loadmat('cuhk03_new_protocol_config_labeled.mat')['camId'].flatten()
labels = loadmat('cuhk03_new_protocol_config_labeled.mat')['labels'].flatten()
train_labels = np.array( loadmat('train_labels.mat')['training_labels'].flatten())
#validlabels = loadmat('valid_labels.mat')['validation_data'].flatten()
training_data = loadmat('train_data_norm.mat')['training_data_norm']
triplet_idx = np.array(loadmat('triplet_idx.mat')['triplet_idx']) # Sets of anchors/positives/negative
n_triplets = len(triplet_idx[0][:])
triplet_idx = triplet_idx.astype(np.float)
query_data = loadmat('query_data.mat')['query_data']
gallery_data = loadmat('gallery_data.mat')['gallery_data']

#testlabels = loadmat('test_labels.mat')['test_labels'].flatten()


#training_data = np.zeros(len(features[1][:]), len(train_idxs))
    
training_data = torch.tensor(training_data) # Define as a tensor - all data that enters the network has to be represented as a tensor
training_data.to(device) # Sendto GPU (if available)
training_data.float() # Cast as a float for compatability

gallery_data = torch.tensor(gallery_data)
gallery_data.float()

query_data = torch.tensor(query_data)
query_data.float()

val_gal = torch.tensor(val_gal)
val_gal.float()

val_que = torch.tensor(val_que)
val_que.float()

#%%

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2048,2048) # Feature Extraction - Fully Connected
       # self.drop = nn.Dropout(0.05)   # Dropout layer to combat overfitting - not implemented 
        self.bn = nn.BatchNorm1d(2048)  # Feature Extraction - BatchNorm
        self.relu = nn.ReLU()           # Feature Extraction - Rectified Linear Unit
        self.fc2 = nn.Linear(2048,2048) # Distance Metric - L - Fully Connected
        
        with torch.no_grad():
            self.fc1.weight.normal_(0,0.001) # Initisalise weights according to normal dist
            self.fc2.weight.normal_(0,0.001)
            
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
#%% VALIDATION
print('Start of Training')
import torch.optim as optim # Loop across hyperparamaters; margin size, weight decay and batch size
margins = [1]              #
weight_decays = [0.001]  # - Adjust these values for hyperparametric sweep
batch_sizes = [500]     #

for margin in margins:   
    for w in weight_decays:
        for batch_size in batch_sizes:
            filename_V_q = "V_q_bno%i_mar_%i_wd%f.txt" % (batch_size, margin,w) # Set file names according to hyperparameters
            filename_V_g = "V_g_bno%i_mar%i_wd%f.txt" % (batch_size, margin,w)
            filename_V_nn = "V_nn_bn%i_mar%i_wd%f.txt" % (batch_size, margin,w)
            filename_V_nn_avg = "V_nn_avg_bn%i_mar%i_wd%f.txt" % (batch_size, margin,w)
            filename_T_q = "T_q_bn%i_mar%i_wd%f.txt" % (batch_size, margin,w)
            filename_T_g = "T_g_bn%i_mar%i_wd%f.txt" % (batch_size, margin,w)
            filename_T_nn = "T_NEW_nn_bn%i_mar%i_wd%f.txt" % (batch_size, margin,w)
            filename_T_nn_avg = "T_nn_avg_bn%i_mar%i_wd%f.txt" % (batch_size, margin,w)
            filename_loss = "loss_bn%i_mar%i_wd%f.txt" % (batch_size, margin,w)
            
            net = Net() # Make and init neural net
            net.to(device) # Send net to GPU
            criterion = TripletLoss(margin).to(device) # Make loss function based on Margin hyperparameter and send to GPU
            optimizer = optim.Adadelta(net.parameters(),weight_decay=w) # Set AdaDelta optimiser based on weight decay hyperparameter
            loss_track = []
            for epoch in range(1):  # loop over the dataset multiple times
                
                for i in range(n_triplets//batch_size): # Number of batches based on batch size hyperparameter
                    # get the inputs
                    in1 = training_data[triplet_idx[0,(i)*batch_size:(i+1)*batch_size]-1].to(device).float() # Anchor, takes slices of total triplets
                                                                                                             # of size batch_size
                    in2 = training_data[triplet_idx[1,(i)*batch_size:(i+1)*batch_size]-1].to(device).float() # Positives
                   
                    in3 = training_data[triplet_idx[2,(i)*batch_size:(i+1)*batch_size]-1].to(device).float() # Negatives
                 
                    # zero the parameter gradients
                    optimizer.zero_grad() # Zero the gradient of the optimiser so it does not accumulate gradient across batches 
                    # forward + backward + optimize
                    output1 = net(in1) # Pushs batches of anchors, positives and negatives through net 
                    output2 = net(in2)
                    output3 = net(in3)
                    
                    loss = criterion(output1,output2,output3) # Find triplet loss for batch
                    loss.backward() # Backprop - find delta W
                    optimizer.step() # Update weights based on delta W 
                    loss_temp = loss.detach().cpu().numpy() # Save loss a a number, not a tensor
                    loss_temp = float(loss_temp)
                    loss_track.append(loss_temp) # Keep track of loss over the training process
                    
                  #  print(loss)
             
                         
            #Projections
            print('Finished Training')
            np.savetxt(filename_loss, loss_track) # Save loss 
            print('Validation')
       
            #Generate M matrix from weights of final layer (representing L)
            
            L = np.array(net.fc2.weight.detach().cpu()) # Detach final layer of network (L)
            M = np.matmul(np.transpose(L), L) # M = L^T * L
            M_root = np.real(scipy.linalg.sqrtm(M)) # Root M - metric instead of psudeo metric
            # Project validation/test data with M
            
            val_gal = net.fc1(val_gal.float().to(device)) # Push validation /test data through the network
            val_gal = net.bn(val_gal)                     # apart from the final layer, and instead project with M_root to 
            val_gal = net.relu(val_gal)                   # ensure a valid distance metrix
            val_gal_pushed = np.transpose(np.matmul(M_root, np.transpose(val_gal.detach().cpu().numpy())))
            
            val_que = net.fc1(val_que.float().to(device))
            val_que = net.bn(val_que)
            val_que = net.relu(val_que)
            val_quer_pushed = np.transpose(np.matmul(M_root, np.transpose(val_que.detach().cpu().numpy())))
            
            gal = net.fc1(gallery_data.float().to(device))
            gal = net.bn(gal)
            gal = net.relu(gal)
            gal_pushed = np.transpose(np.matmul(M_root, np.transpose(gal.detach().cpu().numpy())))
            
            quer = net.fc1(query_data.float().to(device))
            quer = net.bn(quer)
            quer = net.relu(quer)
            quer_pushed = np.transpose(np.matmul(M_root, np.transpose(quer.detach().cpu().numpy())))
            
            
            #VALIDATION - kNN 
            
            K = 10
            possible_quer = np.zeros((val_quer_pushed.shape[0],K)) # initalise
            Labels = np.zeros((val_quer_pushed.shape[0],K))        #
            for i in range(0,val_quer_pushed.shape[0]):
            	deleted_cam_person = np.where(np.logical_not(np.logical_and(val_gal_labels == val_query_labels[i],camId_v_gal == camId_v_quer[i]))==1)[0] # delete same-person
            	nbrs = NearestNeighbors(10,algorithm = 'kd_tree').fit(val_gal_pushed[deleted_cam_person])
            	distances, indices = nbrs.kneighbors([val_quer_pushed[i]])
            	reduced = val_gal_labels[deleted_cam_person]
            	Labels[i] = reduced[indices][:][0].flatten()
            	if i %10 == 0:
            		print(int(i))
            
            results_knn = np.logical_not(Labels-val_query_labels)
            for i in range(0,val_quer_pushed.shape[0]):
            	correct_seen = 0;
            	for j in range(0,K):
            		if results_knn[i][j] == 1:
            			correct_seen += 1
            			results_knn[i][j:10] = 1
            
            			
            results_knn = np.sum(results_knn,axis = 0)/val_quer_pushed.shape[0]
            print(str(results_knn))
            np.savetxt(filename_V_nn, results_knn)
            
            #TESTING - kNN 
            
            Labels = []
    
            possible_quer = np.zeros((quer_pushed.shape[0],K)) 
            Labels = np.zeros((quer_pushed.shape[0],K))    
            for i in range(0,quer_pushed.shape[0]):
            	deleted_cam_person = np.where(np.logical_not(np.logical_and(gallery_labels == query_labels[i],camId_gal == camId_quer[i]))==1)[0]
            	nbrs = NearestNeighbors(10,algorithm = 'kd_tree').fit(gal_pushed[deleted_cam_person])
            	distances, indices = nbrs.kneighbors([quer_pushed[i]])
            	reduced = gallery_labels[deleted_cam_person]
            	Labels[i] = reduced[indices][:][0].flatten()
            	if i %10 == 0:
            		print(int(i))
            
            results_knn = np.logical_not(Labels-query_labels)
            for i in range(0,quer_pushed.shape[0]):
            	correct_seen = 0;
            	for j in range(0,K):
            		if results_knn[i][j] == 1:
            			correct_seen += 1
            			results_knn[i][j:10] = 1
            
            			
            results_knn = np.sum(results_knn,axis = 0)/quer_pushed.shape[0]
            print(str(results_knn))
            np.savetxt(filename_T_nn, results_knn)

            
        
#%% TESTING


