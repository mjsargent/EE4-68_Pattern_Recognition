%% determine optimal number of subspaces

% CW Part 2
% images are 46x56 pixels, 10 images per person
clc;clear;close all
rng(1)
nTrials = 10;
load('face(1).mat');
maxMajTrack = zeros(10,10);
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

%PCA of entire training set
N = 364;
Sl = (1/N)*A'*A;
[Vl,Dl] = eigs(Sl,364);
Vl = normc(A*(Vl));
St = Sw + Sb;
%[Wpca Dpca] = eigs(Vl,312);
%% Feature randomisation: nFixed vectors are the same for all spaces, nSub = number of feature spaces
Mpca = 200;
nFixed = 30;
Mlda = 30;
nSub = 20;

% Compute normal Fisher LDA as baseline to compare to ensemble (here,
% Mpca = 312, Mlda = 51
[Wlda Dlda] = eigs(inv(Vl(:,1:200)'*Sw*Vl(:,1:200))*Vl(:,1:200)'*Sb*Vl(:,1:200),30);      
Wopt = (Wlda'*Vl(:,1:200)')';
proj_train = (trainingData-meanImage)' * Wopt;
proj_test = (testingData-meanImage)' * Wopt;
Idx = knnsearch(proj_train, proj_test);
predicted_class = ITraining(Idx);
results = abs(ITesting - predicted_class);
baselineTrack = sum(~logical(results))/length(ITesting)*100
%%
% loop changing the number of fixed eigenvectors (getting an idea of
% optimal Mpca)

% output plot: plot(1) = majority voting performance
%              plot(2) = best feature space performance in ensemble
%              plot(3) = performance of baseline fisher lda


a = 5;c = 50;
maxSingle = zeros(c/a,1);
maxMaj = zeros(c/a,1);
i = 1;
for trial = 1:10
    trial
    i = 1;
for nSub = a:a:c
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
    accuracy = zeros(nSub+1,1);
    predicted_class = zeros(nSub,156);
    results = zeros(nSub,156);
    for sub = 1:nSub
        tempWpca = reshape(randWpcas(sub,:,:),[2576 Mpca]);
        [Wlda Dlda] = eigs(inv(tempWpca'*Sw*tempWpca)*(tempWpca'*Sb*tempWpca),Mlda);
        Wopt = (Wlda'*tempWpca')';
        proj_train = (trainingData-meanImage)' * Wopt;
        proj_test = (testingData-meanImage)' * Wopt;
        Idx = knnsearch(proj_train, proj_test);
        predicted_class(sub,:) = ITraining(Idx);
        results = abs(ITesting - predicted_class(sub,:));
        accuracy(sub,1) = sum(~logical(results))/length(ITesting)*100;
    end
    % then determine performance of majority voting
    maj_voting = mode(predicted_class,1);
    results = abs(ITesting - maj_voting);
    maxMajTrack(trial,i) = sum(~logical(results))/length(ITesting)*100
   
    % determine best performing feature space 
    %maxSingle(i) = max(accuracy(1:nSub,1));

    
    i = i + 1;
end
end
%%

figure;hold on; plot(5:5:50,mean(maxMajTrack,1));plot(5:5:50,baselineTrack*ones(1,length(maxSingle)))
set(gca,'FontSize',28);xlim([5 50]);xticks([5,10:10:50]);ylim([76 90])
xlabel('No. Models','Interpreter','Latex','fontsize',40);ylabel('Success Rate ($\%$)','Interpreter','Latex','fontsize',40)
legend('Maj','Baseline','Interpreter','Latex','fontsize',35,'Location','East')
print -depsc noModelsFeatRand
