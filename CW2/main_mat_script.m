%% PR CW2
clc;clear;close all
rng(1)
load('./PR_data/cuhk03_new_protocol_config_labeled.mat')
features = jsondecode(fileread('./PR_data/feature_data.json'));

%% Establish training data, validation data, test data
training_data = features(train_idx,:);training_labels = labels(train_idx);
query_data = features(query_idx,:); query_labels = labels(query_idx);
gallery_data = features(gallery_idx,:); gallery_labels = labels(gallery_idx);
training_camId = camId(train_idx);query_camId = camId(query_idx);gallery_camId = camId(gallery_idx);

%  Form the validation set
training_ids = unique(training_labels);
n_val_ids = 100;
validation_ids = sort(training_ids(randperm(length(training_ids),n_val_ids)));
[~, mA] = ismember(validation_ids,training_labels);
[~, mB] = ismember(validation_ids,flipud(training_labels));
mB = length(training_labels) - mB + 1;
cum_sum_validation = [0, cumsum(mB+1-mA)'];
validation_data = zeros(sum(mB+1-mA),size(training_data,2));
validation_labels = zeros(sum(mB+1-mA),1);
validation_idx = zeros(sum(mB+1-mA),1);
for i = 1:length(mB)
    validation_data(cum_sum_validation(i)+1:cum_sum_validation(i+1),:) = training_data(mA(i):mB(i),:);
    validation_labels(cum_sum_validation(i)+1:cum_sum_validation(i+1)) = training_labels(mA(i):mB(i));
    validation_idx(cum_sum_validation(i)+1:cum_sum_validation(i+1)) = train_idx(mA(i):mB(i));
end

% Remove validation images from training set
train_mat = zeros(7368,n_val_ids);
for i = 1:n_val_ids
    train_mat(:,i) = training_labels == validation_ids(i);
end
train_mat = ~(sum(train_mat,2));
training_data = training_data(train_mat,:);
training_labels = training_labels(train_mat);
train_idx = train_idx(train_mat);
training_ids = unique(training_labels);

% form validation query and gallery
n_val_query = 100;
val_gallery = validation_data;
val_gallery_label = validation_labels;
val_query = zeros(n_val_query,2048);
val_query_label = zeros(n_val_query,1);
val_quer_camId = zeros(n_val_query,1);
val_taken_idx = zeros(size(val_gallery,1),1);
query_possible = zeros(n_val_query,10);
k = 0;
for person = validation_ids'
    person_idx = find(validation_labels == person);
    image_choice = randperm(sum(validation_labels == person),1);
    image = val_gallery(person_idx(image_choice),:);
    k = k+1;
    val_query(k,:) = image;
    val_query_label(k) = person;
    val_quer_camId(k) = camId(validation_idx(person_idx(image_choice)));
    val_taken_idx(person_idx(image_choice)) = 1;
    max = sum(and(validation_labels == person,camId(validation_idx) ~= camId(validation_idx(person_idx(image_choice)))));
    query_possible(k,1:max) = 1:max;
    query_possible(k,max+1:10) = max;
end

val_gallery = val_gallery(~logical(val_taken_idx),:);
val_gallery_label = val_gallery_label(~logical(val_taken_idx));
val_gal_camId = camId(validation_idx);
val_gal_camId = val_gal_camId(~logical(val_taken_idx));
training_data_norm = zeros(size(training_data));
 
% Normalise data to unit norm (to be used by network)
for i = 1:size(training_data,1)
    training_data_norm(i,:) = training_data(i,:)/norm(training_data(i,:));
end
for i = 1:size(gallery_data,1)
    gallery_data(i,:) = gallery_data(i,:)/norm(gallery_data(i,:));
end
for i = 1:size(query_data,1)
    query_data(i,:) = query_data(i,:)/norm(query_data(i,:));
end
for i = 1:size(val_gallery,1)
    val_gallery(i,:) = val_gallery(i,:)/norm(val_gallery(i,:));
end
for i = 1:size(val_query,1)
    val_query(i,:) = val_query(i,:)/norm(val_query(i,:));
end
save('val_gal_camId','val_gal_camId');
save('val_quer_camId','val_quer_camId');
save('val_gallery','val_gallery');
save('val_gallery_label','val_gallery_label');
save('val_query','val_query');
save('val_query_label','val_query_label');
save('train_idx','train_idx');
save('train_labels','training_labels');
save('train_data_norm','training_data_norm');
save('query_data','query_data');
save('gallery_data','gallery_data');
save('valid_data','validation_data');
save('valid_labels','validation_data');
save('valid_idx','validation_idx');
save('training_data','training_data')
save('training_labels','training_labels')
save('training_camId','training_camId')
save('query_data','query_data')
save('query_labels','query_labels')
save('gallery_data','gallery_data')
save('gallery_labels','gallery_labels')
save('query_camId','query_camId')
save('gallery_camId','gallery_camId')
save('query_possible','query_possible');
%% generate triplets to be used to train network
camId_train = camId(train_idx);
size_ts = size(unique(training_labels),1);
triplet_count = 0;n_triplets = 1000000;
triplet_idx = zeros(1,3*n_triplets);
while(triplet_count < n_triplets) % triplets of the form: xi,xj,xk
    % choose 60 ids
    ids = training_ids(randperm(size_ts,60));
    trip_c = 0;
    while(trip_c < 5000)
        % choose a random ID
        id = ids(randperm(60,1));
        % determine where in labels this ID exists
        same_id = training_labels == id;
        temp_list = find(same_id);
        % choose a random image from this list to form xi
        person_1 = temp_list(randperm(size(temp_list,1),1));
        % choose xj randomly according to set rules
        diff_cam = camId_train ~= camId_train(person_1);
        p_2_logic = find(and(same_id, diff_cam));
        if size(p_2_logic,1) > 0
            person_2 = p_2_logic(randperm(size(p_2_logic,1),1));
            % choose xk randomly according to set rules
            p_3_logic = find(~same_id);
            person_3 = p_3_logic(randperm(size(p_3_logic,1),1));
            trip_c = trip_c+1;
            triplet_count = triplet_count + 1;
            triplet_idx((triplet_count-1)*3+1:(triplet_count-1)*3+3) = [person_1;person_2;person_3];
        end
    end
end
triplet_idx = reshape(triplet_idx,[3,n_triplets]);
save('triplet_idx','triplet_idx');
%% baseline kNN performance
K = 10; Idx = zeros(size(query_data,1),K);Labels = zeros(size(query_data,1),K);
% perform kNN for each query image in gallery set
for i = 1:size(query_data,1) 
    % find indices that do NOT correspond to same cam & same person
    deleted_cam_person = find(not(and(gallery_labels == query_labels(i),camId(gallery_idx) == camId(query_idx(i)))));
    Idx(i,:) = knnsearch(gallery_data(deleted_cam_person,:),query_data(i,:),'K',K);
    % ensure the returned Idx is used on the same "cropped" gallery set as
    % was searched in
    reduced = gallery_labels(deleted_cam_person);
    Labels(i,:) = reduced(Idx(i,:));
end
% determine correct matches
results_knn = ~logical(Labels-query_labels);
% implement the definition of retrieval accuracy
for i = 1:size(results_knn,1)
    for j = 1:size(results_knn,2)
        if results_knn(i,j) == 1
            results_knn(i,j:size(results_knn,2)) = 1;
        end
    end
end
results_knn = sum(results_knn,1)/size(query_data,1);
save('results_knn_baseline','results_knn');
%% baseline k-means
K = 10; Idx = zeros(size(query_data,1),K);
gal_id_length = length(unique(gallery_labels));Labels = zeros(size(query_data,1),K);
for i = 1:size(query_data,1)
    deleted_cam_person = find(not(and(gallery_labels == query_labels(i),camId(gallery_idx) == camId(query_idx(i)))));
    [~, c_loc] = kmeans(gallery_data(deleted_cam_person,:),gal_id_length/2);
    Idx_temp = knnsearch(c_loc,query_data(i,:),'K',1);
    Idx(i,:) = knnsearch(gallery_data(deleted_cam_person,:),c_loc(Idx_temp,:),'K',K);
    reduced = gallery_labels(deleted_cam_person);
    Labels(i,:) = reduced(Idx(i,:));
end
results_knn = ~logical(Labels-query_labels);
for i = 1:size(results_knn,1)
    for j = 1:size(results_knn,2)
        if results_knn(i,j) == 1
            results_knn(i,j:size(results_knn,2)) = 1;
        end
    end
end
results_knn = sum(results_knn,1)/size(query_data,1)
save('results_kmeans_baseline','results_knn')

%% validation distribution plot
rng(50)
A = val_gallery - mean(val_gallery,1);
N = size(val_gallery,1);
S = (1/N)*(A'*A);
[V,D] = eigs(S,3);
val_gal_proj = (val_gallery-mean(val_gallery,1))*V;

figure;hold on;
x = randperm(size(unique(val_gallery_label),1),7);
for i = x
    person = unique(val_gallery_label);
    person = person(i);
    person_is = find(val_gallery_label == person);
    plot3(val_gal_proj(person_is,1),val_gal_proj(person_is,2),val_gal_proj(person_is,3),'x','MarkerSize',10,'LineWidth',4)
end
grid on;view(25,35);set(gca,'FontSize',20,'Linewidth',1.5)
print('3d_validation','-depsc')

%% gallery distribution plot
rng(50)
A = gallery_data - mean(gallery_data,1);
N = size(gallery_data,1);
S = (1/N)*(A'*A);
[V,D] = eigs(S,3);
val_gal_proj = (gallery_data-mean(gallery_data,1))*V;
x = randperm(size(unique(gallery_labels),1),7);
figure;hold on;
for i = x
    person = unique(gallery_labels);
    person = person(i);
    person_is = find(gallery_labels == person);
    plot3(val_gal_proj(person_is,1),val_gal_proj(person_is,2),val_gal_proj(person_is,3),'x','MarkerSize',10,'LineWidth',4)
end
grid on;view(25,35);set(gca,'FontSize',20,'Linewidth',1.5)
print('3d_gallery','-depsc')

%% PCA investigation on test data
A = gallery_data - mean(gallery_data,1);
N = size(gallery_data,1);
S = (1/N)*(A'*A);
[V,D] = eigs(S,100);
K = 1; Idx = zeros(size(query_data,1),K);Labels = zeros(size(query_data,1),K);
knn_PCA = zeros(10,1);
for j = 1:10
    gal_proj = (gallery_data-mean(gallery_data,1))*V(:,1:j*10);
    quer_proj = (query_data-mean(gallery_data,1))*V(:,1:j*10);
    for i = 1:size(query_data,1)
        deleted_cam_person = find(not(and(gallery_labels == query_labels(i),camId(gallery_idx) == camId(query_idx(i)))));
        Idx(i,:) = knnsearch(gal_proj(deleted_cam_person,:),quer_proj(i,:),'K',K);
        reduced = gallery_labels(deleted_cam_person);
        Labels(i,:) = reduced(Idx(i,:));
    end
    results_knn = ~logical(Labels-query_labels);
    results_knn = sum(results_knn,1)/size(query_data,1);
    knn_PCA(j,1) = results_knn;
end
save('knn_PCA','knn_PCA');
