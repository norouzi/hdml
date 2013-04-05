function data = create_data_hdml(MODE, operand1, operand2, operand3)

data.MODE = MODE;

if (strcmp(MODE, 'sem-mnist') || strcmp(MODE, 'sem-mnist-gpu'))
  % Create Semantic full MNIST
  load('data/mnist-full.mat'); % has rperm_60k inside
  % mnist-full.mat is available at http://www.cs.toronto.edu/~norouzi/research/mlh/data/mnist-full.mat
  
  Ntraining = 60000;
  Ntest = 10000;

  mnist_ltrain = mnist_ltrain(rperm_60k);
  mnist_train = mnist_train(:,rperm_60k);
  
  if (operand1 == 0)
    data.learn_mean = zeros(size(mnist_train,1),1);
  else
    data.learn_mean = mean(mnist_train,2);
  end
  
  data.Ntraining = Ntraining;
  data.Ntest = Ntest;
  data.Ltraining = mnist_ltrain;
  data.Xtraining = mnist_train;
  data.Ltest = mnist_ltest;
  data.Xtest = mnist_test;
  data.MODE = MODE;
  data.labels = unique(data.Ltraining(:));
  if any(data.labels ~= (0:max(data.labels))')
    error('labels should be zero based and continuous.');
  end
  data.nnlabels = zeros(numel(data.labels),1);
  for i = 1:numel(data.labels(:))
    data.nnlabels(i) = sum(data.Ltraining(:) ~= data.labels(i));
  end
  data.indnlabels = zeros(max(data.nnlabels), numel(data.labels));
  for i = 1:numel(data.labels(:))
    data.indnlabels(1:data.nnlabels(i),i) = find(data.Ltraining(:) ~= data.labels(i));
  end  
  clear mnist_train mnist_test mnist_ltrain mnist_ltest;
  
  if (operand1 == 0)
    data.scale = 1/255;
  else
    data.scale = mean(sqrt(1./ ( double(mean(single(data.Xtraining(:)).^2)) - ... % /numel(data.Xtraining(:)) - ...
				 double((mean(data.learn_mean)).^2 )) ));
  end
  
  data.Xtraining = bsxfun(@minus, double(data.Xtraining), data.learn_mean) * data.scale;
  data.Xtest = bsxfun(@minus, double(data.Xtest), data.learn_mean) * data.scale;

  data.Xtraining = single(data.Xtraining);
  data.Xtest = single(data.Xtest);

  data.Straining = sparse(false(Ntraining, Ntraining));
  for i = 0:9
    data.Straining(data.Ltraining==i, data.Ltraining==i) = 1;
  end
  data.StestTraining = sparse(false(Ntest, Ntraining));
  for i = 0:9
    data.StestTraining(data.Ltest == i, data.Ltraining == i) = 1;
  end
  
  if (strcmp(MODE(end-2:end), 'gpu'))
    data.Xtraining = gsingle(data.Xtraining);
    data.Xtest = gsingle(data.Xtest);
    data.Ltraining = (data.Ltraining);
    data.Ltest = (data.Ltest);
  end
  
elseif (strcmp(MODE, 'cifar-10') || strcmp(MODE, 'cifar-10-gpu') || strcmp(MODE, 'cifar-10-GPU'))
  fprintf('The given mode is not supported yet.\n');
else
  fprintf('The given mode is not recognized.\n');
end


function data = construct_data(Xtraining, Xtest, sizeSets, avgNNeighbors, proportionNeighbors, data)

% either avgNNeighbors or proportionNeighbors should be set. The other value should be empty ie., []
% avgNNeighbors is a number which determines the average number of neighbors for each data point
% proportionNeighbors is between 0 and 1 which determines the fraction of [similar pairs / total pairs]

[Ntraining, Ntest] = deal(sizeSets(1), sizeSets(2));
DtrueTraining = distMat(Xtraining);
fprintf('DtrueTraining is done.\n');

if (~isempty(avgNNeighbors))
  sortedD = sort(DtrueTraining, 2);
  threshDist = mean(sortedD(:,avgNNeighbors));
  data.avgNNeighbors = avgNNeighbors;
else
  sortedD = sort(DtrueTraining(:));
  threshDist = sortedD(ceil(proportionNeighbors * numel(sortedD)));
  data.proportionNeighbors = proportionNeighbors;
end

DtrueTestTraining = distMat(Xtest, Xtraining); % size = [Ntest x Ntraining]
fprintf('DtrueTestTraining is done.\n');

data.Xtraining = Xtraining;
data.Xtest = Xtest;  
data.Straining = DtrueTraining < threshDist;
data.StestTraining = DtrueTestTraining < threshDist;

data.Ntraining = Ntraining;
data.Ntest = Ntest;
data.threshDist = threshDist;
data.Dtraining = DtrueTraining;
data.DtestTraining = DtrueTestTraining;

% data.scale = mean(sqrt(1./(sum(data.Xtraining.^2,2)/size(data.Xtraining,2))));
% data.Xtraining = data.Xtraining * data.scale;
% data.Xtest = data.Xtest * data.scale;

