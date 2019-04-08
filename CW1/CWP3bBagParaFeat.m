% CWP3b Randomised subspace parameters 

% images are 46x56 pixels, 10 images per person
clc;clear;close all
%rng(1)

load('face(1).mat');


% generate vector of class labesl for training set 
ITraining = zeros(1,364);
tracker = 0;
for i = 1:364
if (mod((i-1),7) == 0)
        tracker = tracker +1;
end
ITraining(i) = tracker;    
end

% generate vector of class labels for testing set 
ITesting = zeros(1,156);
tracker = 0;
for i = 1:156
if (mod((i-1),3) == 0)
        tracker = tracker +1;
end
ITesting(i) = tracker;    
end

% iterate through each person, randomly selecting 7 images 
randomSampleStore = zeros(52,10);
for person = 1:52
    randomSample = randperm(10,10)+(person-1)*10;
    randomSampleStore(person,:) = randomSample;
end

% group testing and training data
trainingData = zeros(2576,364);
testingData = zeros(2576,156);
for person = 1:52
    for face = 1:7
        i = (person-1)*7 + face;
        trainingData(:,i) = X(:,randomSampleStore(person,face));
    end
    for face = 8:10
        i = (person-1)*3 + (face-7);
        testingData(:,i) = X(:,randomSampleStore(person,face));
    end
end

% determine mean image for person
meanImageStore = zeros(2576,52);
trainingImagesMinusMeans = zeros(2576,364);
for person = 1:52
    meanImageStore(:,person) = mean(trainingData(:,(person-1)*7+1:person*7),2);
    for face = 1:7
        trainingImagesMinusMeans(:,(person-1)*7+face) = trainingData(:,(person-1)*7 + face) - meanImageStore(:,person);
    end
end

% mean image of all training data
meanImage = mean(trainingData,2);

% form A matrix from all training data
A = zeros(2576,364);
for person = 1:52
    for face = 1:7
        ind = (person-1)*7 + face;
        A(:,ind) = X(:,randomSampleStore(person,face)) - meanImage;
    end    
end

% determine Sw & Sb
Sis = zeros(52,2576,2576);
for person = 1:52
    Sis(person,:,:) = trainingImagesMinusMeans(:,(person-1)*7+1:person*7)*trainingImagesMinusMeans(:,(person-1)*7+1:person*7)';
end
Sw = reshape(sum(Sis,1),[2576,2576]);
Sb = (meanImageStore-meanImage)*(meanImageStore-meanImage)';

% PCA of entire training set
N = 364;
Sl = (1/N)*A'*A;
[Vl,Dl] = eigs(Sl,364);
Vl = normc(A*(Vl));

%% Feature randomisation: nFixed vectors are the same for all spaces, nSub = number of feature spaces

% Compute normal Fisher LDA as baseline to compare to ensemble (here,
% Mpca = 312, Mlda = 51
[Wlda Dlda] = eigs((Vl(:,1:312)'*Sw*Vl(:,1:312))\Vl(:,1:312)'*Sb*Vl(:,1:312),51);      
Wopt = (Wlda'*Vl(:,1:312)')';
proj_train = (trainingData-meanImage)' * Wopt;
proj_test = (testingData-meanImage)' * Wopt;
Idx = knnsearch(proj_train, proj_test);
predicted_class = ITraining(Idx);
results = abs(ITesting - predicted_class);
baselineFisherFace = sum(~logical(results))/length(ITesting)*100;

% loop changing the number of fixed eigenvectors (getting an idea of
% optimal Mpca)

% output plot: plot(1) = majority voting performance
%              plot(2) = best feature space performance in ensemble
%              plot(3) = performance of baseline fisher lda

%% randomised parameter subspace
nSub = 20;
Mlda = 30;
a = 20;c = 20;

accuracy = zeros(2*nSub,1);
predicted_class = zeros(3*nSub,156);
PXC = zeros(3*nSub,156);
% generate the random feature spaces
for sub = 1:nSub
    Mlda = randperm(51,1);
    Mpca = randperm(312-Mlda,1)+Mlda;
    Wpca = zeros(2576,Mpca);
    Wpca(:,1:Mpca) = Vl(:,1:Mpca);
    
    % determine accuracies for each feature space
    [Wlda Dlda] = eigs((Wpca'*Sw*Wpca)\(Wpca'*Sb*Wpca),Mlda);
    Wopt = (Wlda'*Wpca')';
    proj_train = (trainingData-meanImage)' * Wopt;
    proj_test = (testingData-meanImage)' * Wopt;
    Idx = knnsearch(proj_train, proj_test);
    predicted_class(sub,:) = ITraining(Idx);
    results = abs(ITesting - predicted_class(sub,:));
    accuracy(sub,1) = sum(~logical(results))/length(ITesting)*100; 
    
    for pred = 1:156
        predicted = predicted_class(sub,pred);
        predictedMean = meanImageStore(:,predicted);
        Wki = Wopt'*(predictedMean-meanImage);
        Wkx = Wopt'*(testingData(:,pred)-meanImage);
        PXC(sub,pred) = (1+(Wkx'*Wki)/(norm(Wkx)*norm(Wki)))/2;
    end
end
classSumPXC = zeros(52,156);
for sub = 1:nSub
    for testFace = 1:156
        predicted = predicted_class(sub,testFace);
        classSumPXC(predicted,testFace) = classSumPXC(predicted,testFace) + PXC(sub,testFace);
    end
end
sum_voting = zeros(4,156);[maxPXC,sum_voting(1,:)] = max(classSumPXC);
sum_voting_acc = zeros(4,1);
maj_voting = zeros(4,156);maj_voting_acc = zeros(4,1);

maj_voting(1,:) = mode(predicted_class(1:nSub,:),1);
results = abs(ITesting - maj_voting(1,:));
maj_voting_acc(1,1) = sum(~logical(results))/length(ITesting)*100;

results = abs(ITesting - sum_voting(1,:));
sum_voting_acc(1,1) = sum(~logical(results))/length(ITesting)*100;

figure;plot(ones(10,1)*maj_voting_acc(1,1));hold on;plot(ones(10,1)*sum_voting_acc(1,1))
legend('maj','sum')




%% randomised feature space
Mpca = 150;
nFixed = 30;
Mlda = 15;
nSub = 20;

% generate the random feature spaces
randWpcas = zeros(nSub,2576,Mpca);
for sub = 1:nSub
    % fixed vectors
    randWpcas(sub,:,1:nFixed) = Vl(:,1:nFixed);
    % random vectors
    randEigVecs = randperm(363-nFixed,Mpca-nFixed) + nFixed;
    for vec = nFixed+1:Mpca
        randWpcas(sub,:,vec) = Vl(:,randEigVecs(vec-nFixed));
    end
end
% determine accuracies for each feature space
for sub = 1:nSub
    tempWpca = reshape(randWpcas(sub,:,:),[2576 Mpca]);
    [Wlda Dlda] = eigs((tempWpca'*Sw*tempWpca)\(tempWpca'*Sb*tempWpca),Mlda);
    Wopt = (Wlda'*tempWpca')';
    proj_train = (trainingData-meanImage)' * Wopt;
    proj_test = (testingData-meanImage)' * Wopt;
    Idx = knnsearch(proj_train, proj_test);
    predicted_class(sub+nSub,:) = ITraining(Idx);
    results = abs(ITesting - predicted_class(sub,:));
    accuracy(sub+nSub,1) = sum(~logical(results))/length(ITesting)*100;
    
    for pred = 1:156
        predicted = predicted_class(sub+nSub,pred);
        predictedMean = meanImageStore(:,predicted);
        Wki = Wopt'*(predictedMean-meanImage);
        Wkx = Wopt'*(testingData(:,pred)-meanImage);
        PXC(sub+nSub,pred) = (1+(Wkx'*Wki)/(norm(Wkx)*norm(Wki)))/2;
    end
end
classSumPXC = zeros(52,156);
for sub = 1:nSub
    for testFace = 1:156
        predicted = predicted_class(sub+nSub,testFace);
        classSumPXC(predicted,testFace) = classSumPXC(predicted,testFace) + PXC(sub+nSub,testFace);
    end
end
[maxPXC,sum_voting(2,:)] = max(classSumPXC);
results = abs(ITesting - sum_voting(2,:));
sum_voting_acc(2,1) = sum(~logical(results))/length(ITesting)*100;
% then determine performance of majority voting
maj_voting(2,:) = mode(predicted_class(nSub+1:2*nSub,:),1);
results = abs(ITesting - maj_voting(2,:));
maj_voting_acc(2,1) = sum(~logical(results))/length(ITesting)*100;

figure;plot(ones(10,1)*maj_voting_acc(2,1));hold on;plot(ones(10,1)*sum_voting_acc(2,1))
legend('maj','sum')

%% bagging 

% apply PCA to X_train
% find avg face and differential faces for test and train
X_train_avg = meanImageStore;
X_train_diff = trainingData - meanImage;
X_test_diff = testingData - meanImage;

A = X_train_diff;
N_people = 52;train = 7; test = 3;
M_pca = train * N_people - 1;
% generate num_eigs low dim eigs
S_lowdim = A' * A;
[V_lowdim_all,~] = eigs(S_lowdim, M_pca);

% turn into vector of eigenvalues
% D_lowdim = sum(D_lowdim, 1);

% generate U, the high dim eigenvectors (?)
U_all = A * V_lowdim_all;
U_all = normc(U_all);

% project train, test onto U
X_train_proj = U_all' * X_train_diff;
X_test_proj = U_all' * X_test_diff;

% set number of bags, number of samples per bag
N_bags = 30;
N_samples = 250;

indexes = randi(train * N_people, [N_bags, N_samples]);
indexes = reshape(indexes, [1, N_bags * N_samples]);

samples = X_train_proj(:, indexes);
samples = reshape(samples, [M_pca, N_bags, N_samples]);

sample_labels = ITraining(indexes);
sample_labels = reshape(sample_labels, [N_bags, N_samples]);

% test_image = mat2gray(reshape(samples(:, 11, 79), [height, width]));
% imshow(test_image, 'InitialMagnification', 'fit');

S_W_all_bags = zeros(N_bags, M_pca, M_pca);
mean_class_images = zeros(N_bags, M_pca, N_people);

W_opt_all_bags = zeros(N_bags, M_pca, N_people - 1);

for bag = 1:N_bags
    current_bag_labels = sample_labels(bag, :);
    classes = sort(unique(current_bag_labels));
    for class = classes
        indexes = find(current_bag_labels == class);
        current_train = reshape(samples(:, bag, indexes), [M_pca, size(indexes,2)]);

        mean_class_image = mean(current_train, 2);
        mean_class_images(bag, :, class) = mean_class_image;

        diffed_class_image = current_train - mean_class_image;
        S_W_all_bags(bag, :,:) = squeeze(S_W_all_bags(bag, :,:)) + diffed_class_image * diffed_class_image';
    end
    
    S_W = squeeze(S_W_all_bags(bag, :,:));
    
    mean_train_image = mean(squeeze(samples(:, bag, :)),2);
    diffed_class_mean_images = squeeze(mean_class_images(bag, :, :)) - mean_train_image;
    S_B = (diffed_class_mean_images)*(diffed_class_mean_images)';

    S_T = S_B + S_W;

%     M_pca = 312;
    M_pca_bag = rank(S_W);

    [W_pca, ~] = eigs(S_T, M_pca_bag);

    intermediate = (W_pca' * S_W * W_pca)\(W_pca' * S_B * W_pca);

%     M_lda = 51;
    M_lda_bag = rank(S_B);

    [W_lda, ~] = eigs(intermediate, M_lda_bag);
    
    W_opt = zeros(M_pca, N_people - 1);

    W_opt(:, 1:M_lda_bag) = W_pca * W_lda;
    
    W_opt_all_bags(bag,:,:) = W_opt;
end

accuracies_all_bags = zeros(1, N_bags);
predicted_classes_all_bags = zeros(N_bags, N_people * test);

for bag = 1:N_bags
    W_opt = squeeze(W_opt_all_bags(bag, :,:));
    
    proj_train = X_train_proj' * W_opt;
    proj_test = X_test_proj' * W_opt;

    Idx = knnsearch(proj_train, proj_test);

    % use l to determine if we got the right person, results(n) = 0 if we did
    predicted_class(2*nSub+bag,:) = ITraining(Idx);    
    results = abs(ITesting - predicted_class(2*nSub+bag,:));
    
    %predicted_classes_all_bags(bag, :) = predicted_class;

    % every time we get it wrong, make it a positive number, then turn it
    % into a logical array, invert and sum it to get the number of correct
    % test examples, divide by the total number of tests and turn into a
    % percentage
    accuracies_all_bags(bag) = sum(~logical(results))/length(ITesting)*100;
    
    for pred = 1:156
        predicted = predicted_class(bag+2*nSub,pred);
        predictedMean = meanImageStore(:,predicted);
        Wki = Wopt'*(predictedMean-meanImage);
        Wkx = Wopt'*(testingData(:,pred)-meanImage);
        PXC(bag+2*nSub,pred) = (1+(Wkx'*Wki)/(norm(Wkx)*norm(Wki)))/2;
    end
end
classSumPXC = zeros(52,156);

for sub = 1:nSub
    for testFace = 1:156
        predicted = predicted_class(sub+2*nSub,testFace);
        classSumPXC(predicted,testFace) = classSumPXC(predicted,testFace) + PXC(sub+2*nSub,testFace);
    end
end
[maxPXC,sum_voting(3,:)] = max(classSumPXC);
results = abs(ITesting - sum_voting(3,:));
sum_voting_acc(3,1) = sum(~logical(results))/length(ITesting)*100;

results_maj_vote_bags = abs(ITesting - mode(predicted_class(2*nSub+1:3*nSub,:)));
maj_voting_acc(3,1) = sum(~logical(results_maj_vote_bags))/length(ITesting)*100;

figure;plot(ones(10,1)*maj_voting_acc(3,1));hold on;plot(ones(10,1)*sum_voting_acc(3,1))
legend('maj','sum')
%% all

maj_voting(4,:) = mode(predicted_class(1:3*nSub,:),1);
results = abs(ITesting - maj_voting(4,:));
maj_voting_acc(4,1) = sum(~logical(results))/length(ITesting)*100;



classSumPXC = zeros(52,156);

for sub = 1:3*nSub
    for testFace = 1:156
        predicted = predicted_class(sub,testFace);
        classSumPXC(predicted,testFace) = classSumPXC(predicted,testFace) + PXC(sub,testFace);
    end
end
[maxPXC,sum_voting(4,:)] = max(classSumPXC);
results = abs(ITesting - sum_voting(4,:));
sum_voting_acc(4,1) = sum(~logical(results))/length(ITesting)*100;

figure;plot(ones(10,1)*maj_voting_acc(4,1));hold on;plot(ones(10,1)*sum_voting_acc(4,1))
legend('maj','sum')


%% combined performance plot

figure;subplot(1,2,1);hold on;plot(baselineFisherFace*ones(1,20));
plot(maj_voting_acc(1,1)*ones(1,20));plot(maj_voting_acc(2,1)*ones(1,20));plot(maj_voting_acc(3,1)*ones(1,20));
plot(maj_voting_acc(4,1)*ones(1,20))
legend('baselin fisher','rand param space','rand feature space','bagging','combined param, feature and bagging')
subplot(1,2,2);hold on;plot(baselineFisherFace*ones(1,20));
plot(sum_voting_acc(1,1)*ones(1,20));plot(sum_voting_acc(2,1)*ones(1,20));plot(sum_voting_acc(3,1)*ones(1,20));
plot(sum_voting_acc(4,1)*ones(1,20))

legend('baselin fisher','rand param space','rand feature space','bagging','combined param, feature and bagging')