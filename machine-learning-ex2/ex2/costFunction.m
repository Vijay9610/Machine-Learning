function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
temp = 0;
for i=1:m
  temp = temp - [y(i,1)*log(sigmoid(theta(1,1)+theta(2,1)*X(i,2)+theta(3,1)*X(i,3))) + (1-y(i,1))*log(1-sigmoid(theta(1,1)+theta(2,1)*X(i,2)+theta(3,1)*X(i,3)))];
endfor
J = 1/m * temp;

temp1 = 0;
for i=1:m
  temp1 = temp1 + (sigmoid(theta(1,1) + theta(2,1) * X(i,2) + theta(3,1) * X(i,3)) - y(i,1))*X(i,1);
endfor
grad(1) = 1/m * temp1;

temp2 = 0;
for i=1:m
  temp2 = temp2 + (sigmoid(theta(1,1) + theta(2,1) * X(i,2) + theta(3,1) * X(i,3)) - y(i,1))*X(i,2);
endfor
grad(2) = 1/m * temp2;

temp3 = 0;
for i=1:m
  temp3 = temp3 + (sigmoid(theta(1,1) + theta(2,1) * X(i,2) + theta(3,1) * X(i,3)) - y(i,1))*X(i,3);
endfor
grad(3) = 1/m * temp3;


% =============================================================

end
