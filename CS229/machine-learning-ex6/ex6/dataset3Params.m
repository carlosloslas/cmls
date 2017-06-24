function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

parameter_no = 2;
parameter_range = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
parameter_combinations = zeros(length(parameter_range)^parameter_no, parameter_no + 1);
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
min_error = 100;

for i = 1:length(parameter_range)
  for j = 1:length(parameter_range)
    % initialise temporary C & sigma parameters
    c_temp = parameter_range(i);
    sigma_temp = parameter_range(j);
    
    % modelTrain call from ex6.m
    % model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
    model= svmTrain(X, y, c_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp)); 
    
    % make predictions & calculate prediction errors on the cross-validation dataset
    predictions = svmPredict(model, Xval);
    prediction_error = mean(double(predictions ~= yval));
    
    %% add error, c_temp & sigma_temp to the parameter combinations
    %parameter_combinations(i*j, :) = [prediction_error, c_temp, sigma_temp+;
    % after for loops, check which one is the minimum error... a bit too 
    % complex without pythonic indexing.
    
    if prediction_error < min_error
      min_error = prediction_error;
      C = c_temp;
      sigma = sigma_temp;
    endif  
  end
end

% =========================================================================

end
