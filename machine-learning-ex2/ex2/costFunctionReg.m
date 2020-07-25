function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

n = length(theta);

%temp = 0;
%for i=1:m
%  temp = temp - [y(i,1)*log(sigmoid(theta(1,1)+theta(2,1)*X(i,2)+theta(3,1)*X(i,3))) + (1-y(i,1))*log(1-sigmoid(theta(1,1)+theta(2,1)*X(i,2)+theta(3,1)*X(i,3)))];
%endfor
temp = log(sigmoid(theta' * X')) * y + log(1-sigmoid(theta' * X')) * (1-y);

temp_reg = 0;
for j=2:n
  temp_reg = temp_reg + theta(j,1)*theta(j,1);
endfor
J = -1 * 1/m * temp + lambda/(2*m) * temp_reg;
grad(1) = 1/m*(sigmoid(theta' * X') - y')*X(:,1);

for j=2:n
  grad(j) = 1/m * (sigmoid(theta' * X') - y')*X(:,j) + lambda/m * theta(j);
endfor


% =============================================================

end
