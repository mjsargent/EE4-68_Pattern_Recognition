% CW Part 1
% images are 46x56 pixels, 10 images per person
clc;clear;close all

load('face(1).mat')
rng(1);

%% Q1a -------------------------------------------------------------
% Apply PCA to training data, compute eigenvectors and eigenvalues of
% the covariance matrix S = (1/N)AA^T
% N = number of images, A = matrix with columns (xn - xbar, dim = DxN)

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

% calculate mean image
meanImage = zeros(2576,1);
for person = 1:52
    for face = 1:7
        meanImage = meanImage + X(:,randomSampleStore(person,face));
    end
end
meanImage = meanImage / 364;

% form the A matrix by subtracting mean image from all test images
A = zeros(2576,364);
for person = 1:52
    for face = 1:7
        ind = (person-1)*7 + face;
        A(:,ind) = X(:,randomSampleStore(person,face)) - meanImage;
    end    
end

% form covariance matrix, S
N = 364;
S = (1/N)*A*A';

% output mean image
meanImageMatrix = zeros(56,46);
for i = 1:46
     meanImageMatrix(:,i) = meanImage(1+(i-1)*56:i*56,1);
end
figure(1);imshow(mat2gray(meanImageMatrix),'InitialMagnification','fit');
colormap(gray(256));print -depsc meanImage

%% obtain eigenvalues/vectors, visualise eigenvalue values
[V,D] = eigs(S,2576);
eigValueVector = zeros(2576,1);
for i = 1:2576
    eigValueVector(i) = D(i,i);
end
eigValueVector = flipud(sort(eigValueVector));
figure(2);plot(1:363,abs(eigValueVector(1:363)));set(gca,'FontSize',18);
xticks([1,50,100,150,200,250,300,350])
ylabel('Magnitude of Eigenvalue','Interpreter','Latex','fontsize',23);
xlabel('Principal Component Number','Interpreter','Latex','fontsize',23);
xlim([1 363])
print -depsc sEigValues

% display first five eigenfaces
fiveEigenFaces = zeros(56,46,5);
for n = 1:5
    for i = 1:46
        fiveEigenFaces(:,i,n) = V(1+(i-1)*56:i*56,n);
    end
end
figure;imshow(mat2gray(fiveEigenFaces(:,:,1)),'InitialMagnification','fit');
colormap(gray(256));print -depsc l1Eigenface
figure;imshow(mat2gray(fiveEigenFaces(:,:,2)),'InitialMagnification','fit');
colormap(gray(256));print -depsc l2Eigenface




%% find how many eigenvalues are "non-zero"
for i = 1:2576
    if abs(eigValueVector(i)) < 0.1
        disp(sprintf('There are %d "non-zero" eigenvalues.',i-1))
        break
    end  
end              
% explanation as to why number of non-zero components is at most N-1
% https://stats.stackexchange.com/questions/123318/why-are-there-only-n-1-principal-components-for-n-data-if-the-number-of-dime

%% create cumulative percentage variance plot
cumVariancePercent = cumsum(abs(eigValueVector(1:363)))/sum(abs(eigValueVector(1:363)));
figure;plot(1:363,0.95*ones(363),'r--',1:363,cumVariancePercent,'b');ylabel('Percentage of total Variance');xlabel('Number of Principal Components Used')
xlim([0,363]);legend('P = 95%','Location','SouthEast')
print -depsc cumVariancePlot


%% determine how many components needed to retain 95% of total variance
for i = 1:363
    if cumVariancePercent(i) >= 0.95
        disp(sprintf("%d principal components are needed to retain 95 percent of total variance.",i))
        break
    end
end

%% Q1b: Low-dimensional computation of eigenspace 

% determine low-dimensional eigenvalues/vectors, compute difference between
% these and higher dimensional computation
Sl = (1/N)*A'*A;
[Vl,Dl] = eigs(Sl,364);
eigValueVectorL =  zeros(364,1);
for i = 1:364
    eigValueVectorL(i) = Dl(i,i);
end
Vl = normc(Vl);
eigValueVectorL = flipud(sort(eigValueVectorL));

VReconstructed = A*(Vl);
VReconstructed = normc(VReconstructed);% some eigvectors are flipped in direction 

diffHighLow = eigValueVectorL - eigValueVector(1:364);

cumReconstructionError = zeros(1,364);
reconstructionErrorByPerson = zeros(52,364);
projectionsByM = zeros(364,364,364);
reconstructionErrorByFaceM = zeros(364,7,52); % for finding face with highest / lowest recon error
for M = 1:364
    mVReconstructed = VReconstructed(:,1:M);
    for person = 1:52
        cumReconstructionByPerson = 0;
        for i = 1:7
            currentFace = X(:,randomSampleStore(person,i)); 
            normCurrentFace = currentFace - meanImage; % normalise 
            a = normCurrentFace'*mVReconstructed; % determine projection coefficient vector
            reconstructedFace = meanImage + mVReconstructed*a'; % reconstruct
            differenceVector = reconstructedFace - currentFace; % difference vector
            normVec = vecnorm(differenceVector)^2;
            reconstructionErrorByFaceM(M,i,person) = normVec;
            cumReconstructionError(M) = cumReconstructionError(M) + normVec; % find mag of difference vector
            cumReconstructionByPerson = cumReconstructionByPerson + normVec;
            % (the following is used for NN)
            projectionsByM((7*(person-1) + i),1:M,M) = a; %projectionsByM(faceNumber, coeffic number, number of faces used)
        end
        reconstructionErrorByPerson(person,M) = cumReconstructionByPerson;
    end
end
figure;plot(cumReconstructionError);
set(gca,'FontSize',18);
xticks([1,50,100,150,200,250,300,350])
xlabel('PCA Bases Learned','Interpreter','Latex','fontsize',23);
ylabel('Reconstruction Error','Interpreter','Latex','fontsize',23);
xlim([1,363]);print -depsc cumReconstructionError

%% determine face(s) with highest/lowest reconstruction error
sumReconByFace = sum(reconstructionErrorByFaceM,1);
[Mar,Iar] = max(sumReconByFace(:));
[Mir,Iir] = min(sumReconByFace(:));
personMax = ceil(Iar/7); faceMax = Iar - (personMax-1)*7;
personMin = ceil(Iir/7); faceMin = Iir - (personMin-1)*7;

%plot min & max:
figure;
plot(1:364,reconstructionErrorByFaceM(:,faceMin,personMin),...
     1:364,reconstructionErrorByFaceM(:,faceMax,personMax));
legend('min recon error','max recon error');print -depsc lowAndHighReconFaces

%display min and max images
lowReconFace = zeros(56,46);highReconFace = zeros(56,46);
for i = 1:46
    lowReconFace(:,i) = X(1+(i-1)*56:i*56,randomSampleStore(personMin,faceMin)); 
    highReconFace(:,i) = X(1+(i-1)*56:i*56,randomSampleStore(personMax,faceMax)); 
end

figure;imshow(mat2gray(lowReconFace),'InitialMagnification','fit')
colormap(gray(256));print -depsc lowReconErrorFace
figure;imshow(mat2gray(highReconFace),'InitialMagnification','fit')
colormap(gray(256));print -depsc highReconErrorFace

%% NN Classifer

predictedByM = zeros(156,364);
percentageCorrect = zeros(1,364);
for M = 1:364
    predictedCorrectCount = 0;
    for person = 1:52
        for i = 1:3
            testFace = X(:,randomSampleStore(person,(i+7))) - meanImage; %normalise
            testProjection = testFace'*VReconstructed(:,1:M); % project test image into eigenspace
            bestError = 100000000000; % arbitrary large value
            for j = 1:364
                error = norm((testProjection - projectionsByM(j,1:M,M))); %retrieve all the eigenspace coefficents for a given projection and M
                if((error < bestError))
                    bestError = error;
                    predictedFace = j;
                    predictedFaceClass = ITraining(predictedFace); %Retrieve predicted class from best projection - need variable
                end
            end
            predictedByM((3*(person-1) + i),M) = predictedFaceClass;
            if (predictedFaceClass == person)
                predictedCorrectCount = predictedCorrectCount + 1;
            end
        end
    end
    percentageCorrect(M) = predictedCorrectCount/156;
end

% confusion matrix for M = 1,2,5,10,20,50,100,200,364

con = confusionmat(ITesting, predictedByM(:,5));
figure;confusionchart(con);%title([' ', num2str(i),' PCA Base(s) Learnt']);
print -depsc confusion5NN

con = confusionmat(ITesting, predictedByM(:,100));
figure;con1 = confusionchart(con);%title([' ', num2str(i),' PCA Base(s) Learnt']);
set(gca,'FontSize',23)
%xlabel(con1,'fontsize',23)
%ylabel('True Class','Interpreter','Latex','fontsize',23)
xticks(5:5:50)
print -depsc confusion100NN


figure;plot(percentageCorrect*100);
set(gca,'FontSize',18);
xticks([1,50,100,150,200,250,300,350])
xlabel('PCA Bases Learned','Interpreter','Latex','fontsize',23);ylabel('Success Rate ($\%$)','Interpreter','Latex','fontsize',23);
xlim([1 363])
print -depsc successRateNN




%% Reconstruction error classification (subspace for each class)
    meanImageStore = zeros(2576,52);
    Ar = zeros(2576,7);
    VrStore = zeros(2576,7,52);

% created subspaces for each class
for person = 1:52
    % determine mean image for person
    for face = 1:7
        meanImageStore(:,person) = meanImageStore(:,person) + X(:,randomSampleStore(person,face));
    end
    meanImageStore(:,person) = meanImageStore(:,person)/7;
    % determine A matrix for each person
    for face = 1:7
        Ar(:,face) = X(:,randomSampleStore(person,face)) - meanImageStore(:,person);
    end
    
    Sr = (1/7)*Ar'*Ar;
    [Vr,Dr] = eigs(Sr,7);
    VrStore(:,:,person) = normc(Ar*Vr);
end

altPercentageCorrect = zeros(7,1);
estimatedFaces = zeros(7,52,3);
normVec = zeros(7,52,3,52);
for M = 1:7
    predictedCorrectCount = 0;
    for person = 1:52
        for face = 8:10
            currentLowestError = 100000000000000;
            currentFace = X(:,randomSampleStore(person,face));           
            for recon = 1:52
                testProject = (currentFace-meanImageStore(:,recon))'*VrStore(:,1:M,recon);
                reconCurrentFace = meanImageStore(:,recon) + VrStore(:,1:M,recon)*testProject';
                diffVector = currentFace - reconCurrentFace;
                normVec(M,person,face-7,recon) = norm(diffVector)^2;
                if normVec(M,person,face-7,recon) < currentLowestError
                    currentLowestError = normVec(M,person,face-7,recon);
                    estimatedFaces(M,person,face-7) = recon;
                end
            end
            if estimatedFaces(M,person,face-7) == person
                predictedCorrectCount = predictedCorrectCount + 1;
            end
        end
    end
    altPercentageCorrect(M) = predictedCorrectCount*100/156;
end
figure;plot(altPercentageCorrect);xlim([1 6]);
set(gca,'FontSize',18);
xlabel('PCA Bases Learned','Interpreter','Latex','fontsize',23);
ylabel('Success Rate ($\%$)','Interpreter','Latex','fontsize',23);
print -depsc reconstructionPercCorrect


%% reshape estimated face to be MxN for use in confusion matrix
predictedByRecM = zeros(156,7);
for M = 1:7
    for person = 1:52
        predictedByRecM((((3*(person-1)) + 1):((3*(person-1))) + 3),M) = estimatedFaces(M,person,:);
    end
end
 
%% confusion matrix for M = 1:7
for i = 6
    con = confusionmat(ITesting, predictedByRecM(:,i));
    figure;confusionchart(con);
    print -depsc confusionReconstruction

end


%% PCA analysis NN scatter for succ and unsucc
% calculate mean image

projectionMatrix = zeros(10,2);
person = 2;
for j = 1:7
    projectionMatrix(j,:) = A(:,(person-1)*7+j)'*VReconstructed(:,1:2);
end
for j = 8:10
    projectionMatrix(j,:) = (X(:,randomSampleStore(person,j))-meanImage)'*VReconstructed(:,1:2);
end
figure;
for j = 1:7
    hold on;
    if j == 2
         plot(projectionMatrix(8,1),projectionMatrix(8,2),'bx')
    end
    plot(projectionMatrix(j,1),projectionMatrix(j,2),'rx')
    xlim([-3000 3000]);ylim([-3000 3000]);
end
for j = 9:10
    hold on;
    plot(projectionMatrix(j,1),projectionMatrix(j,2),'bx')
    xlim([-3000 3000]);ylim([-3000 3000]);
end

set(gca,'FontSize',28);
xlabel('Coordinate 1','Interpreter','Latex','fontsize',40);
ylabel('Coordinate 2','Interpreter','Latex','fontsize',40);
yticks([-2000 0 2000])
print -depsc unsuccessfulScatterNN


person = 4;
for j = 1:7
    projectionMatrix(j,:) = A(:,(person-1)*7+j)'*VReconstructed(:,1:2);
end
for j = 8:10
    projectionMatrix(j,:) = (X(:,randomSampleStore(person,j))-meanImage)'*VReconstructed(:,1:2);
end
figure;
for j = 1:7
    hold on;
    if j == 2
        plot(projectionMatrix(8,1),projectionMatrix(8,2),'bx')
    end
    plot(projectionMatrix(j,1),projectionMatrix(j,2),'rx')
    xlim([-3000 3000]);ylim([-3000 3000]);
end
for j = 9:10
    hold on;
    plot(projectionMatrix(j,1),projectionMatrix(j,2),'bx')
    xlim([-3000 3000]);ylim([-3000 3000]);
end

set(gca,'FontSize',28);
xlabel('Coordinate 1','Interpreter','Latex','fontsize',40);
ylabel('Coordinate 2','Interpreter','Latex','fontsize',40);
yticks([-2000 0 2000])
legend('Training Data','Test Data','Interpreter','Latex','fontsize',40)
print -depsc successfulScatterNN

successVariance = sum(var(X(:,31:40)'))
unSuccessVariance = sum(var(X(:,11:20)'))
%%








    %std(projectionMatrix)
%perhaps a graph showing correlation between variance and success rate?
% %% determine success rate for each class over all M
% percentCorrectOverMByClass = zeros(52,1);
% projectionMatrix = zeros(7,2);
% 
% for person = 1:52
%     predictedCorr = 0;
%     for M = 1:10
%         for face = 1:3
%             if predictedByM((3*(person-1) + face),M) == person
%                 predictedCorr = predictedCorr + 1;
%             end
%         end
%     end
%     percentCorrectOverMByClass(person) = predictedCorr/(3*10);
% end
% 
% varianceByPerson = zeros(52,1);
% centroidByPerson = zeros(52,364);
% projectionMatrix = zeros(7,364);
% for person = 1:52
%     centroid = 0;
%     for j = 1:7
%         temp = A(:,(person-1)*7+j)'*Vrs(:,1:364);
%         projectionMatrix(j,:) = temp;
%         centroid = centroid + temp;
%     end
%     centroidByPerson(person,:) = centroid/7;
%     vTemp = var(projectionMatrix);
%     varianceByPerson(person) = sum(vTemp);
% end
% minDistancePerPerson = zeros(52,1);
% for person = 1:52
%     smallestDistance = 10000000000000000000;
%     centroid = centroidByPerson(person);
%     for i = 1:52
%         if i == person
%             continue
%         end
%         otherCentroid = centroidByPerson(i);
%         distance = norm(centroid-otherCentroid);
%         if distance < smallestDistance
%             smallestDistance = distance;
%         end
%     end
%     minDistancePerPerson(person) = smallestDistance;
% end
% 
% figure;
% for i = 1:52
%     hold on
%     plot(varianceByPerson(i)/minDistancePerPerson(i),percentCorrectOverMByClass(i),'rx')
% end
% 
% 
% 
