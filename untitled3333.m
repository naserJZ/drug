% Read the CSV file
data = readtable('drug200.csv'); % Replace 'drug200.csv' with the path to your CSV file
% Display the data
disp(data);

% Split the data into predictors (X) and target (y)
X = data(:, 1:end-1); % Replace '1:end-1' with the appropriate columns of predictors
yCell = data(:, end); % Replace 'end' with the column index of the target variable
% Convert the target variable y to a categorical array
y = categorical(yCell{:, 1});
% Split the data into training and testing sets
rng(1); % Set random seed for reproducibility
cvp = cvpartition(size(data, 1), 'Holdout', 0.5); % 50% train, 50% test
XTrain = X(cvp.training, :);
yTrain = y(cvp.training, :);
XTest = X(cvp.test, :);
yTest = y(cvp.test, :);
% Build the decision tree classifier
tree = fitctree(XTrain, yTrain);
% Predict using the decision tree
ypred = predict(tree, XTest);

% Evaluate the performance of the decision tree
accuracy = sum(ypred == yTest) / numel(yTest);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);

% View the decision tree
view(tree, 'Mode', 'graph');
