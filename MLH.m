% Copyright (c) 2011, Mohammad Norouzi and David Fleet

function [Wset] = MLH(data_in, loss_cell, nb, eta_set, momentum, size_batches_set, trainset, maxiter_set, ...
		      zerobias, nval_during, nval_after, verbose, shrink_w_set, shrink_eta, ...
		      initW, etabound, regularizer)

% Performs validation on sets of parameters by calling appropriate instances of learnMLH function.
%
% Input:
%    data_in: data structure for training the model made by create_data.m
%    loss_cell: a cell array that determines the type of loss and its parameters. First element of
%      this cell defines loss type; either 'bre', or 'hinge'. For 'hinge' loss two other elements
%      are required defining 'rho' and 'lambda'. Using arrays for 'rho' and 'lamda' results in
%      validation on those parameters. For example, loss_cell being {'hinge', [3 4], [0 .5]} results
%      in validation on rho over 3 and 4, and on lambda over 0 and .5 (totally four configurations).
%    nb: number of bits
%    eta_set: choices for learning rate
%    momentum: momentum parameter for gradient descent (we always use .9)
%    size_batches_set: mini-batch size for gradient descent (we always use 100)
%    trainset: can be either 'train' or 'trainval'. Using 'train' splits the training set into train
%      and validation sets. Using 'trainval' performs training on the complete training set.
%    maxiter: number of iterations
%    zerobias: either 0 or 1, meaning whether the hashing hyper-planes' biases should be all
%      zero or should be learned. Both possibilities can be provided for validation.
%    nval_during: how many validations during training
%    nval_after: how many validation after training (to account for validation noise)
%    verbose: either 0 or 1, writing debug information or not
%
% Output:
%    A structure array storing sets of weight matrices (W), parameters (params), average precision
%      (ap), etc. learned by MLH

if (isa(data_in.Xtraining, 'gdouble') || isa(data_in.Xtraining, 'gsingle'))
  gpu = 1;
elseif (isa(data_in.Xtraining, 'GPUsingle') || isa(data_in.Xtraining, 'GPUdouble'))
  gpu = 2;
else
  gpu = 0;
end

param.initw_sigma = .1;

% if (strcmp(data_in.MODE(1:5), 'sem-m'))
%   param.initw_sigma = .1;
% elseif (strcmp(data_in.MODE(1:5), 'cifar'))
%   param.initw_sigma = .1;
% end

if (numel(nb) > 1)
  if (exist('initW', 'var'))
    param.nonlinearity = 1;
  else
    param.nonlinearity = 3;
  end
else
  param.nonlinearity = NaN;
end

% initialization
if (~exist('initW', 'var') || isempty(initW) || iscell(initW))
  if (numel(nb) == 1)
    fprintf('single-layer initialization\n');
    initW = [param.initw_sigma*randn(nb, size(data_in.Xtraining, 1))]; % LSH
    if (~zerobias)
      initW(:,end) = initW(:,end) - median(initW * data_in.Xtraining, 2);
    else
      initW(:,end) = 0;
    end
    
    % initW = [param.initw_sigma*randn(nb, size(data_in.Xtraining, 1)-1) zeros(nb, 1)]; % LSH
    % initW(:,end) = -median(initW * data_in.Xtraining, 2);

    % initW(:,end) = 0;
    
    % % Normalization on W's
    % normW2 = sqrt(sum(initW.^2,2));
    % initW = bsxfun(@rdivide, initW, 2*normW2);

    % if (data_in.Ntraining > 100000)
    %   if (isa(data_in.Xtraining, 'uint8'))
    % 	x = bsxfun(@minus, double(data_in.Xtraining(:,1:100000)), data_in.learn_mean) * data_in.scale;
    %   else
    % 	x = data_in.Xtraining(:, 1:100000);
    %   end
    % else
    %   if (isa(data_in.Xtraining, 'uint8'))
    % 	x = bsxfun(@minus, double(data_in.Xtraining), data_in.learn_mean) * data_in.scale;
    %   else
    % 	x = data_in.Xtraining;
    %   end
    % end
  else
    param.nonlinearity = 1;
    
    for i=1:numel(nb)
      
      % if (i == 1 && strcmp(data_in.MODE(1:5), 'cifar'))
      % 	param.initw_sigma = .01;
      % else
      % 	param.initw_sigma = .1;
      % end
      
      n1i = nb(i);
      if (i == 1)
	n2i = size(data_in.Xtraining,1);
      else
	n2i = nb(i-1)+1;
      end
      
      if (~exist('initW') || numel(initW)<i || isempty(initW{i}))
	if (i < nb)
	  initW{i} = [param.initw_sigma * randn(n1i, n2i)];
	else
	  initW{i} = [param.initw_sigma * randn(n1i, n2i)];
	end
	
	if (data_in.Ntraining > 10000)
	  if (i > 1)
	    resp = compute_NN_output(initW(1:i-1), [data_in.Xtraining(:, 1:10000)], param.nonlinearity);
	  else
	    resp = {data_in.Xtraining(:, 1:10000)};
	  end
	else
	  if (i > 1)
	    resp = compute_NN_output(initW(1:i-1), [data_in.Xtraining], param.nonlinearity);
	  else
	    resp = {data_in.Xtraining};
	  end
	end

	if (i > 1)
	  initW{i}(:,end) = -median(initW{i}(:,1:end-1)*resp{end},2);
	else
	  initW{i}(:,end) = initW{i}(:,end) - median(initW{i} * resp{end}, 2);
	end
      end
      % initW{i}(:, end) = zeros(n1i, 1);	% offset terms
      
      % if (i == numel(nb))
      % 	% if (data_in.Ntraining > 10000)
      % 	%   resp = compute_NN_output(initW, [data_in.Xtraining(:, 1:10000); ones(1, 10000)]);
      % 	% else
      % 	%   resp = compute_NN_output(initW, [data_in.Xtraining; ones(1, data_in.Ntraining)]);
      % 	% end
      % 	% initW{end}(:,end) = -median(initW{end}(:,1:end-1)*(resp{numel(initW)-1}),2);
      % 	initW{end}(:,end) = 0;
      % end
      
      % Normalization on W's
      % initW{i} = initW{i} ./ [repmat(sqrt(sum(initW{i}.^2,2)), [1 n2i])];
    end
  end
end
nb = nb(end);

if (gpu == 1)
  if (~iscell(initW))
    initW = gdouble(initW);
  else
    for i=1:numel(initW)
      initW{i} = gdouble(initW{i});
    end  
  end
end

if (gpu == 2)
  if (~iscell(initW))
    initW = GPUsingle(initW);
  else
    for i=1:numel(initW)
      initW{i} = GPUsingle(initW{i});
    end  
  end
end

% same initialization is used for comparision of parameters
data = create_training(data_in, trainset, nval_during + nval_after);
% if (strcmp(data.MODE, 'inria-sift-1B'))
%   nn = findNN(data.Xtest, data.Xtraining);
%   data.nn = nn;
% end
if (verbose)
  display(data);
end

losstype = loss_cell{1};
if strcmp(losstype, 'hinge')
  rho_set = loss_cell{2};
  lambda_set = loss_cell{3};
  scale_set = loss_cell{4};
  m = 1;
  for scale = scale_set
    for rho = rho_set
      for lambda = lambda_set
	loss_set(m).type = losstype;
	loss_set(m).rho = rho;
	loss_set(m).lambda = lambda;
	loss_set(m).scale = scale;
	m = m+1;
      end
    end
  end
elseif (strcmp(losstype, 'triplet-ranking') || strcmp(losstype, 'triplet-ranking2'))
  scale_set = loss_cell{2};
  lmargin = loss_cell{3};
  m = 1;
  for scale = scale_set
    loss_set(m).type = losstype;
    loss_set(m).scale = scale;
    loss_set(m).margin = lmargin;
    m = m+1;
  end
else
  loss_set(1).type = losstype;
end

% Wset = 0;
% return;

n = 1;
for maxiter = maxiter_set
for size_batches = size_batches_set
for eta = eta_set
for shrink_w = shrink_w_set
for loss = loss_set
  
  param.noisy = 0;
  param.regularizer = regularizer;
  
  param.mineta = etabound(1);
  param.maxeta = etabound(2);
  param.fast = 1;
  param.bootstrap = 0;
  param.gpu = gpu;
  param.size_batches = size_batches;
  param.loss = loss;
  param.shrink_w = shrink_w;
  param.nb = nb;
  param.eta = eta;
  param.maxiter = maxiter;
  param.momentum = momentum;
  param.zerobias = zerobias;
  param.trainset = trainset;
  param.mode = data.MODE;
  if (isfield(data, 'Ntraining'))
    param.Ntraining = data.Ntraining;
  else
    param.Ntraining1 = data.Ntraining1;
    param.Ntraining2 = data.Ntraining2;
  end
  param.nval_during = nval_during;
  param.nval_after = nval_after;
  param.shrink_eta = shrink_eta;
  
  [ap W Wall params] = learnMLH(data, param, verbose, initW);
  % [ap W Wall params] = learnMLH_triplet(data, param, verbose, initW);
  % [ap W Wall params] = learnMLH_GPU(data, param, verbose, initW);
  
  if (~verbose)
    if (numel(size_batches_set) > 1)
      fprintf('batch-size: %d  ', size_batches);
    end
    if (numel(loss_set) > 1)
      if strcmp(loss.type, 'hinge')
	fprintf('rho:%d / lambda:%.2f  ', loss.rho, loss.lambda);
      end
    end
    if (numel(eta_set) > 1)
      fprintf('eta: %.3f  ', eta);
    end
    if (numel(shrink_w_set) > 1)
      fprintf('shrink_w: %.0d  ', shrink_w);
    end
    fprintf(' --> ap:%.3f\n', ap);
  end
  
  Wset(n).ap = ap;
  Wset(n).W = W;
  Wset(n).params = params;
  Wset(n).mode = data_in.MODE;
  
  % Because PCA is not necessarily unique, we store the prinicipal components of the data with the
  % learned weights too.
  if (isfield(data_in, 'princComp'))
    Wset(n).princComp = data_in.princComp;
  end
  n = n+1;

end
end
end
end
end
