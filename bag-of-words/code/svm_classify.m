% Starter code prepared by James Hays for Computer Vision

%This function will train a linear SVM for every category (i.e. one vs all)
%and then use the learned linear classifiers to predict the category of
%every test image. Every test feature will be evaluated with all 15 SVMs
%and the most confident SVM will "win". Confidence, or distance from the
%margin, is W*X + B where '*' is the inner product or dot product and W and
%B are the learned hyperplane parameters.

function predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)
% image_feats is an N x d matrix, where d is the dimensionality of the
%  feature representation.
% train_labels is an N x 1 cell array, where each entry is a string
%  indicating the ground truth category for each training image.
% test_image_feats is an M x d matrix, where d is the dimensionality of the
%  feature representation. You can assume M = N unless you've modified the
%  starter code.
% predicted_categories is an M x 1 cell array, where each entry is a string
%  indicating the predicted category for each test image.

%{
Useful functions:
 matching_indices = strcmp(string, cell_array_of_strings)
 
  This can tell you which indices in train_labels match a particular
  category. This is useful for creating the binary labels for each SVM
  training task.

[W B] = vl_svmtrain(features, labels, LAMBDA)
  http://www.vlfeat.org/matlab/vl_svmtrain.html

  This function trains linear svms based on training examples, binary
  labels (-1 or 1), and LAMBDA which regularizes the linear classifier
  by encouraging W to be of small magnitude. LAMBDA is a very important
  parameter! You might need to experiment with a wide range of values for
  LAMBDA, e.g. 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10.

  Matlab has a built in SVM, see 'help svmtrain', which is more general,
  but it obfuscates the learned SVM parameters in the case of the linear
  model. This makes it hard to compute "confidences" which are needed for
  one-vs-all classification.

%}

%unique() is used to get the category list from the observed training
%category list. 'categories' will not be in the same order as in proj4.m,
%because unique() sorts them. This shouldn't really matter, though.

% with vocab_size = 200
% LAMBDA        Accuracy
% 10            0.439 0.439 0.439 0.439 0.439 0.439 0.439 0.439 0.439 0.439
% 1             0.332 0.389 0.329 0.335 0.357 0.371 0.349 0.375 0.285 0.372
% 0.1           0.473 0.432 0.437 0.396 0.481 0.439 0.427 0.443 0.419 0.396
% 0.01          0.444 0.466 0.514 0.463 0.539 0.499 0.443 0.530 0.491 0.498
% 0.001         0.584 0.613 0.605 0.616 0.609 0.607 0.609 0.557 0.602 0.604
% 0.0001        0.643 0.656 0.668 0.669 0.660 0.660 0.658 0.663 0.654 0.665
% 0.00001       0.611 0.647 0.652 0.649 0.646 0.647 0.640 0.642 0.651 0.648

% averages
% LAMBDA        Accuracy
% 10            0.4390
% 1             0.3494
% 0.1           0.4343
% 0.01          0.4887
% 0.001         0.6006
% 0.0001        0.6597
% 0.00001       0.6433

categories = unique(train_labels); 
num_categories = length(categories);
LAMBDA = .0001;

ws = zeros(15, size(train_image_feats, 2));
bs = zeros(15, 1);

% make 15 svms
for i=1:num_categories
    category = categories(i);
    binary = double(strcmp(category, train_labels));
    binary(binary == 0) = -1;
    [W B] = vl_svmtrain(train_image_feats', binary, LAMBDA);
    ws(i, :) = W;
    bs(i, :) = B;
end

result = train_labels;

for i=1:size(test_image_feats, 1)
    confidences = dot(ws,...
        repmat(test_image_feats(i, :), num_categories, 1),...
        2) + bs;
    [match, index] = max(confidences);
    result(i) = categories(index);
end

predicted_categories = result;




