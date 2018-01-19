function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%
#{
alpha = 0.01;
theta = zeros(n+1,1);

for i = 1:num_labels
    initial_theta = zeros(n+1, 1); 
    J_history = zeros(50, 1);
    
    %[J,grad] = lrCostFunction(initial_theta, X, (y==i), lambda);
    
    %dimension of grad = 401 x 1
    
    for iter = 1:50
        [J,grad] = lrCostFunction(initial_theta, X, (y==i), lambda);
        for j = 1:size(initial_theta,1) %401   
            theta(j) = theta(j) - (alpha/m)*grad(j);
        end
        hypo = sigmoid(X*theta);
        J_history(iter) = (-1/m) * (y'*log(hypo) + (1 - y)'*log(1 - hypo)) + (lambda/(2*m)) * sum(initial_theta(2:end).^2);      
    end
    
    if i==1                     %where i = class
      all_theta(i,:) = theta;
      figure;
      plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);

      xlabel('Number of iterations');
      ylabel('Cost J');
end

#}


for i = 1:num_labels
    initial_theta = zeros(n + 1, 1);
    options = optimset('GradObj', 'on', 'MaxIter', 50);
    all_theta(i,:) = ...
         fmincg (@(t)(lrCostFunction(t, X, (y == i), lambda)), ...
                 initial_theta, options);



% =========================================================================


end
