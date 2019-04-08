%% All coursework images (bar 3d plots)
clc;clear;close all
%% Baseline Approaches
% kNN & k-means
load('results_kmeans_baseline.mat')
results_kmean = results_knn;
load('results_knn_baseline.mat')

figure;plot(1:10,results_knn,'o-',1:10,results_kmean,'x-','MarkerSize',15)
set(gca,'FontSize',20)
xlabel('Rank','Interpreter','Latex','fontsize',25)
xticks([1:10]);xlim([1,10]);ylim([0 1])
ylabel('Retrieval Accuracy','Interpreter','Latex','fontsize',25)
grid on
legend('kNN','k-means','Location','best',...
                            'Interpreter','Latex','fontsize',25)
print('baseline_ret_error','-depsc')

%% Validation, best performance: b 500, m = 1, w 0.001
%% Load data
load('./PR_CW2_Results/V_nn_bn500_mar1_wd0.001000.txt')
load('./PR_CW2_Results/loss_bn500_mar1_wd0.001000.txt')
load('./PR_CW2_Results/V_nn_bn5000_mar1_wd0.001000.txt')
load('./PR_CW2_Results/loss_bn5000_mar1_wd0.001000.txt')
load('./PR_CW2_Results/V_nn_bn1000_mar1_wd0.001000.txt')
load('./PR_CW2_Results/loss_bn1000_mar1_wd0.001000.txt')
load('./PR_CW2_Results/V_nn_bn500_mar1_wd0.000100.txt')
load('./PR_CW2_Results/loss_bn500_mar1_wd0.000100.txt')
load('./PR_CW2_Results/V_nn_bn500_mar1_wd0.000000.txt')
load('./PR_CW2_Results/loss_bn500_mar1_wd0.000000.txt')
load('./PR_CW2_Results/V_nn_bn500_mar0_wd0.001000.txt')
load('./PR_CW2_Results/loss_bn500_mar0_wd0.001000.txt')
load('./PR_CW2_Results/V_nn_bn500_mar10_wd0.001000.txt')
load('./PR_CW2_Results/loss_bn500_mar10_wd0.001000.txt')
load('./PR_CW2_Results/T_nn_bn1000_mar1_wd0.000100.txt')
load('./PR_CW2_Results/T_nn_bn500_mar1_wd0.001000.txt')
load('./PR_CW2_Results/T_nn_bn500_mar10_wd0.001000.txt')


%% fixed margin, fixed weight decay, changing batch
figure;plot(1:10,V_nn_bn5000_mar1_wd0_001000,'o-',1:10,V_nn_bn1000_mar1_wd0_001000,'x-',1:10,V_nn_bn500_mar1_wd0_001000,'^-',...
    'MarkerSize',20);
set(gca,'FontSize',28)
xlabel('Rank','Interpreter','Latex','fontsize',40)
xticks([1:10]);xlim([1,10])
ylabel('Retrieval Accuracy','Interpreter','Latex','fontsize',40)
grid on
legend('Bn = 5000','Bn = 1000','Bn = 500','Location','best',...
                            'Interpreter','Latex','fontsize',35)
ylim([0 1.1])
print('perf_fixed_m_w_change_b','-depsc')

figure;plot(1000000/size(loss_bn5000_mar1_wd0_001000,1):1000000/size(loss_bn5000_mar1_wd0_001000,1):1000000,loss_bn5000_mar1_wd0_001000,...
            1000000/size(loss_bn1000_mar1_wd0_001000,1):1000000/size(loss_bn1000_mar1_wd0_001000,1):1000000,loss_bn1000_mar1_wd0_001000,...
            1000000/size(loss_bn500_mar1_wd0_001000,1):1000000/size(loss_bn500_mar1_wd0_001000,1):1000000,loss_bn500_mar1_wd0_001000,...
    'MarkerSize',20);
set(gca,'FontSize',28)
xlabel('Triplets Learned ($\times 10^6$)','Interpreter','Latex','fontsize',40)
xticks([500000,1000000]);xticklabels([0.5,1]);
ylabel('Estimated Loss','Interpreter','Latex','fontsize',40)
grid on
legend('Bn = 5000','Bn = 1000','Bn = 500','Location','best',...
                            'Interpreter','Latex','fontsize',35)
print('loss_fixed_m_w_change_b','-depsc')

%% fixed margin, changing weight decay, fixed batch
figure;plot(1:10,V_nn_bn500_mar1_wd0_001000,'o-',1:10,V_nn_bn500_mar1_wd0_000100,'x-',1:10,V_nn_bn500_mar1_wd0_000000,'^-',...
    'MarkerSize',20);
set(gca,'FontSize',28)
xlabel('Rank','Interpreter','Latex','fontsize',40)
xticks([1:10]);xlim([1,10])
ylabel('Retrieval Accuracy','Interpreter','Latex','fontsize',40)
grid on
legend('Wd = 0.0010','Wd = 0.0001','Wd = 0.0000','Location','best',...
                            'Interpreter','Latex','fontsize',35)
ylim([0 1.1])
print('perf_fixed_m_b_change_w','-depsc')

figure;plot(1000000/size(loss_bn500_mar1_wd0_001000,1):1000000/size(loss_bn500_mar1_wd0_001000,1):1000000,loss_bn500_mar1_wd0_001000,...
            1000000/size(loss_bn500_mar1_wd0_000100,1):1000000/size(loss_bn500_mar1_wd0_000100,1):1000000,loss_bn500_mar1_wd0_000100,...
            1000000/size(loss_bn500_mar1_wd0_000000,1):1000000/size(loss_bn500_mar1_wd0_000000,1):1000000,loss_bn500_mar1_wd0_000000,...
    'MarkerSize',20);
set(gca,'FontSize',28)
xlabel('Triplets Learned ($\times 10^6$)','Interpreter','Latex','fontsize',40)
xticks([500000,1000000]);xticklabels([0.5,1]);
ylabel('Estimated Loss','Interpreter','Latex','fontsize',40)
grid on
legend('Wd = 0.0010','Wd = 0.0001','Wd = 0.0000','Location','best',...
                            'Interpreter','Latex','fontsize',35)
print('loss_fixed_m_b_change_w','-depsc')

%% changing margin, fixed weight decay, fixed batch
figure;plot(1:10,V_nn_bn500_mar10_wd0_001000,'o-',1:10,V_nn_bn500_mar1_wd0_001000,'x-',1:10,V_nn_bn500_mar0_wd0_001000,'^-',...
    'MarkerSize',20);
set(gca,'FontSize',28)
xlabel('Rank','Interpreter','Latex','fontsize',40)
xticks([1:10]);xlim([1,10])
ylabel('Retrieval Accuracy','Interpreter','Latex','fontsize',40)
grid on
legend('M = 10','M = 1','M = 0.1','Location','best',...
                            'Interpreter','Latex','fontsize',35)
ylim([0 1.1])
print('perf_fixed_b_w_change_m','-depsc')

figure;plot(1000000/size(loss_bn500_mar10_wd0_001000,1):1000000/size(loss_bn500_mar10_wd0_001000,1):1000000,loss_bn500_mar10_wd0_001000,...
            1000000/size(loss_bn500_mar1_wd0_001000,1):1000000/size(loss_bn500_mar1_wd0_001000,1):1000000,loss_bn500_mar1_wd0_001000,...
            1000000/size(loss_bn500_mar0_wd0_001000,1):1000000/size(loss_bn500_mar0_wd0_001000,1):1000000,loss_bn500_mar0_wd0_001000,...
    'MarkerSize',20);
set(gca,'FontSize',28)
xlabel('Triplets Learned ($\times 10^6$)','Interpreter','Latex','fontsize',40)
xticks([500000,1000000]);xticklabels([0.5,1]);
ylabel('Estimated Loss','Interpreter','Latex','fontsize',40)
grid on
legend('M = 10','M = 1','M = 0.1','Location','best',...
                            'Interpreter','Latex','fontsize',35)
print('loss_fixed_b_w_change_m','-depsc')

%% top 3 performing test combinations
%1: b500 m1 w0.001
%2: b500 m10 w0.001
%3: b1000 m1 w0.0001

figure;plot(1:10,results_knn,'o-',1:10,T_nn_bn500_mar1_wd0_001000,'x-',1:10,T_nn_bn500_mar10_wd0_001000,'^-',1:10,T_nn_bn1000_mar1_wd0_000100,'s-'...
            ,'MarkerSize',15)
set(gca,'FontSize',20)
xlabel('Rank','Interpreter','Latex','fontsize',25)
xticks([1:10]);xlim([1,10]);ylim([0.3 0.75])
ylabel('Retrieval Accuracy','Interpreter','Latex','fontsize',25)
grid on
legend('kNN','500;1;0.0010','500;10;0.0010','1000;1;0.00010','Location','best',...
                            'Interpreter','Latex','fontsize',25)
print('best_test_pknn','-depsc')

%% knn baseline K=1 with PCA
load('./knn_PCA.mat')
figure;plot(10:10:100,knn_PCA,'o-','MarkerSize',15)
set(gca,'FontSize',20)
xlabel('Retained Principal Components','Interpreter','Latex','fontsize',25)
xticks([10:10:100]);xlim([10,100]);ylim([0.3 0.5])
ylabel('Rank 1 Retrieval Accuracy','Interpreter','Latex','fontsize',25)
grid on
legend('kNN','Location','best',...
                            'Interpreter','Latex','fontsize',25)
print('knn_PCA','-depsc')