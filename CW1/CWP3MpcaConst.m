% CW Part 2
% images are 46x56 pixels, 10 images per person
clc;clear;close all
rng(1)
accuracy = zeros(51,1);

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

% generate vector of class labesl for testing set 
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
    
    meanImageStore = zeros(2576,52);
    trainingImagesMinusMeans = zeros(2576,364);
    
    % determine mean image for person
    for person = 1:52
        meanImageStore(:,person) = mean(trainingData(:,(person-1)*7+1:person*7),2);
        for face = 1:7
            trainingImagesMinusMeans(:,(person-1)*7+face) = trainingData(:,(person-1)*7 + face) - meanImageStore(:,person);
        end
    end
    meanImage = mean(trainingData,2);
    Sis = zeros(52,2576,2576);
    for person = 1:52
        Sis(person,:,:) = trainingImagesMinusMeans(:,(person-1)*7+1:person*7)*trainingImagesMinusMeans(:,(person-1)*7+1:person*7)';
    end
    Sw = reshape(sum(Sis,1),[2576,2576]);
    Sb = (meanImageStore-meanImage)*(meanImageStore-meanImage)';
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
%[VSw DSw] = eig(Sw);
%[VSb DSb] = eig(Sb);
Mpca = 150;
St = Sw + Sb;
[Wpca Dpca] = eigs(St,312);

    for Mlda = 1:51
        Mlda
        MpcaTemp = Vl(:,1:Mpca);
        
        [Wlda Dlda] = eigs((MpcaTemp'*Sw*MpcaTemp)\(MpcaTemp'*Sb*MpcaTemp),Mlda);
        
        
        Wopt = (Wlda'*MpcaTemp')';
        
        proj_train = (trainingData-meanImage)' * Wopt;
        proj_test = (testingData-meanImage)' * Wopt;
        
        Idx = knnsearch(proj_train, proj_test);
        
        % use l to determine if we got the right person, results(n) = 0 if we did
        predicted_class = ITraining(Idx);
        results = abs(ITesting - predicted_class);
        
        % every time we get it wrong, make it a positive number, then turn it
        % into a logical array, invert and sum it to get the number of correct
        % test examples, divide by the total number of tests and turn into a
        % percentage
        accuracy(Mlda,1) = sum(~logical(results))/length(ITesting)*100;
    end


figure;plot(accuracy);
set(gca,'FontSize',28);
xlabel('$M_{LDA}$','Interpreter','Latex','fontsize',40);
ylabel('Success Rate $(\%)$','Interpreter','Latex','fontsize',40);
xlim([1 51]);ylim([0 85])
xticks([1,10,20,30,40,50])
print -depsc MldaVaryMpcaConst



