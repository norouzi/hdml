% Copyright (c) 2011-2013, Mohammad Norouzi and David Fleet

function [mean_ap final_W Wall final_params] = learnMLH(data, param, verbose, initW)

% The main file for learning hash functions. It performs stochastic
% gradient descent to learn the hash parameters. It is an improvement
% on top of the original MLH code to enable the use of triplets and
% neural networks.
%
% Input:
%    data: data structure for training the model already split into training and validation sets.
%    param: a parameter structure which should include the required parameters.
%    verbose: either 0 or a positive number i, writing debug information every i epochs or not
%    initW: initial weight matrix. initW is a cell array for neural nets
%
% Output:
%    mean_ap: mean average precision over nval_after validation stages after training
%    final_W: final weight matrix
%    Wall: a set of weight matrices (or cells) stored during training at intermediate validation stages
%    final_params: parameters with some additional components

if (~isinf(data.nn))
  nnTraining = data.nnTraining;
  nns = data.nns;
end
if (isfield(data,'nn2') && ~isinf(data.nn2))
  nnTraining2 = zeros(data.nn2, data.Ntraining);
  whos nnTraining2
end

tic_id_full = tic;
nb = param.nb;				% number of bits i.e, binary code length
initeta = param.eta;			% initial learning rate
shrink_eta = param.shrink_eta;		% whether shrink learning rate, as training proceeds
size_batches = param.size_batches;	% mini-batch size
maxiter = param.maxiter;		% number of gradient update iterations (each iteration
                                        % consists of 10^5 pairs)
zerobias = param.zerobias;		% whether offset terms are learned for hashing hyper-planes
                                        % or they all go through the origin
momentum = param.momentum;		% momentum term (between 0 and 1) for gradient update
shrink_w = param.shrink_w;		% weight decay parameter

loss_func = param.loss;			% loss_function is a structure itself. The code supports
                                        % loss_func.type = 'triplet-ranking', 'hinge', 'bre'.  For
					% hinge loss provide two other parameters 'loss_func.rho'
                                        % and 'loss_func.lambda'. For bre loss no parameter is
                                        % needed.

if strcmp(loss_func.type, 'bre')
  %% Most likely doesn't work!
  % Normalizing distance values for bre
  data.Dtraining = data.Dtraining / max(data.Dtraining(:));
end
if (isfield(data, 'Ntraining'))
  Ntraining1 = data.Ntraining;
  Ntraining2 = data.Ntraining;
  NtrainingSqr = data.Ntraining^2;
else
  Ntraining1 = data.Ntraining1;
  Ntraining2 = data.Ntraining2;
  NtrainingSqr = Ntraining1 * Ntraining2;
end  
Xtraining = data.Xtraining;

if strcmp(loss_func.type, 'hinge')
  lambda = loss_func.lambda;
  if numel(loss_func.rho) == 1
    rho1 = loss_func.rho;
    rho2 = loss_func.rho;
  else
    rho1 = loss_func.rho(1);
    rho2 = loss_func.rho(2);
  end
  rho = round((rho1 + rho2) / 2);
else
  rho = NaN;
end

if (param.gpu == 2)
  y1p = GPUsingle(zeros(nb, size_batches, 'single'));		% (y1p, y2p) are the solutions to loss-adjusted inference
  y2p = GPUsingle(zeros(nb, size_batches, 'single'));
  y3p = GPUsingle(zeros(nb, size_batches, 'single'));
end
tval = floor(maxiter/(param.nval_during));

if (verbose)
  fprintf('---------------------------\n');
  fprintf('nb = %d\n', nb);
  fprintf('losstype = ''%s''', loss_func.type);
  if strcmp(loss_func.type, 'hinge')
    fprintf(', rho1/rho2 = %d/%d', rho1, rho2);
    fprintf(', lambda = %.3f', lambda);
    fprintf(', scale = %.2f', loss_func.scale);
  elseif (strcmp(loss_func.type, 'triplet-ranking') || strcmp(loss_func.type, 'triplet-ranking2'))
    fprintf(', scale = %.2f', loss_func.scale);
    fprintf(', margin = %.1f', loss_func.margin);
  end
  fprintf('\n');
  if (zerobias)
    fprintf('zero offset = yes\n');
  else
    fprintf('zero offset = no\n');
  end
  fprintf('weight decay(s) = '); fprintf('''%.0d'' ', shrink_w); fprintf('\n');
  fprintf('max iter = %d\n', maxiter);
  fprintf('init eta = %.1d\n', initeta);
  if (param.shrink_eta == 1)
    fprintf('shrink eta = linear\n');
  elseif (param.shrink_eta == 2)
    fprintf('shrink eta = bold driver\n');
  else
    fprintf('shrink eta = no\n');
  end
  fprintf('size mini-batches = %d\n', size_batches);
  fprintf('momentum = ''%.2f''\n', momentum);
  fprintf('validation during / after training = %d / %d times\n', param.nval_during, param.nval_after);
  % more that one validation after training is suggested and averaging to account for validation noise
  fprintf('[mineta maxeta] = [%.1d %.1d]\n', param.mineta, param.maxeta);
  fprintf('gpu = %d\n', param.gpu);
  if (param.regularizer)
    fprintf('regularizer = %.2f\n', param.regularizer);
  else
    fprintf('regularizer = no\n');
  end
  fprintf('verbose = %d\n', verbose);
  fprintf('nns = [%d %d %d %d]\n', data.nns(1:4)');
  fprintf('---------------------------\n');
end

input_dim = size(Xtraining, 1);
W = initW;

% last2_bound = inf;
last_bound = inf;
last_emploss = inf;
% last2_emploss = inf;

bound = 0;
emploss = 0;
do_break = 0;

if (verbose && (param.nval_during || param.nval_after))
  eval(data, W, param, rho, verbose);
end
if (verbose)
  if (~iscell(W))
    normW = .5 * sum(sum(W.^2));
  else
    normW = 0;
    for i=1:numel(W)
      normW = normW + .5 * sum(sum(W{i}.^2));
    end
  end
  
  fprintf('norm W=%.2f\n', double(normW));
end

% initialization
ntraining = data.Ntraining; 		% total number of pairs to be considered in each iteration
ncases = size_batches;
maxb = floor(data.Ntraining / ncases);	% number of mini-batches
if (maxb ~= ntraining/ncases)
  fprintf('** Ntraining is not divisible by ncases **\n');
end
maxt = maxiter+param.nval_after-1;	% number of epochs

casespos = zeros(maxb*ncases,1);
casesneg = zeros(maxb*ncases,1);
npos = 0;
nneg = 0;

mean_ap  = 0;
avg_acc = Inf;
nnz = 0;
fracs = [0 0];

if (~iscell(W))
  if (param.gpu == 0)
    Winc = zeros(size(W));
  elseif (param.gpu == 1)
    Winc = gzeros(size(W), 'gdouble');
  elseif (param.gpu == 2)
    Winc = GPUsingle(Winc);
  end
else
  Winc = cell(numel(W), 1);
  for i=1:numel(W)
    Winc{i} = zeros(size(W{i}));
    if (param.gpu == 2)
      Winc{i} = GPUsingle(Winc{i});
    end
  end
end

cases = zeros(ncases,1);
x1nd = zeros(ncases,1);
x2nd = zeros(ncases,1);
if (strcmp(loss_func.type, 'triplet-ranking'))
  x3nd = zeros(ncases,1);
end
Wall = cell(maxt+1,1);
Wall{1} = W;
eta = initeta;

if (strcmp(loss_func.type, 'hinge'))
  ncases2 = min(round(ncases * lambda), ncases); 
  ncases1 = ncases - ncases2;
  
  l = zeros(1,ncases);
  l(ncases1+1:end) = 1;
  norm1 = loss_func.scale / (nb-rho1+1);
  norm2 = loss_func.scale / (rho2+1);
  
  loss = kron(l==1, ([zeros(1, rho1) 1:(nb-rho1+1)] * norm1)') + ...
	 kron(l==0, ([(rho2+1):-1:1 zeros(1, nb-rho2)] * norm2)');
  
  rep1tonb = repmat((1:nb)',[1 ncases]);
  rep0tonc = repmat(0:nb:(ncases-1)*nb,[nb, 1]);
  
  if (param.gpu == 1)
    loss = gsingle(loss);
    rep1tonb = gsingle(rep1tonb);
    rep0tonc = gsingle(rep0tonc);
  end
elseif (strcmp(loss_func.type, 'triplet-ranking'))
  loss_margin = loss_func.margin;
  loss = ([zeros(nb-loss_margin, 1); (1:(nb+1+loss_margin))'] * ones(1,ncases)) * loss_func.scale;
elseif (strcmp(loss_func.type, 'triplet-ranking2'))
  loss_margin = loss_func.margin;
  loss = ([zeros(nb-loss_margin, 1); (1:(nb+1+loss_margin)).^2'] * ones(1,ncases)) * loss_func.scale;
end

time_val_tic = tic;
for t=1:maxt
  % Update the target neighbors  
  if (param.bootstrap && exist('nnTraining', 'var'))    
    fprintf('.');
    if (~iscell(W))
      B1 = W*[Xtraining];
    else
      resp = compute_NN_output(W, [Xtraining], param.nonlinearity);
      B1 = resp{end};
      clear resp
    end
    
    B1 = logical(single(B1 > 0));
    B1 = compactbit(B1);

    nl = zeros(numel(data.labels(:)), 1);
    for l = data.labels(:)'
      nl(l+1) = sum(data.Ltraining == l);
    end
    
    for l = data.labels(:)'
      which_l = data.Ltraining == l;
      id_l = find(which_l);
      nn_l = nl(l+1)-1;
      % nns(which_l) = nn_l;
      [b nnTraining_l c d] = linscan_sorted_multi_mex(B1(:,which_l), B1(:,which_l), sum(which_l), ...
						      nb, ceil(nb/2), nn_l+1, 4);      
      for i=1:numel(id_l)
  	nnTraining(1:nn_l,id_l(i)) = id_l(nnTraining_l(2:end, i));
      end
    end
  end
  
  if (t <= min(25,maxt/20))
    mementum = param.momentum/2;
  else
    momentum = param.momentum;
  end
  % ONLY for neural nets
  if (t >= min(50,maxt/10) && iscell(W))
    momentum = .95;
  end
  
  if (shrink_eta == 1)			% learning rate update
    eta = initeta * (maxt-t)/(maxt);
  end
  n_bits_on    = 0;
  n_bits_total = 0;
  tic_id_iter = tic;

  
  % selecting mini-batches for this iteration:
  M = maxb * ncases;
  many_perm = ceil(M/Ntraining1);
  sbatch1 = zeros(many_perm*Ntraining1,1);
  for i=1:many_perm;
    sbatch1((i-1)*Ntraining1 + (1:Ntraining1)) = randperm(Ntraining1);
  end
  [sbatch2 sbatch3] = select_neighbors(sbatch1, data);
  
  % if strcmp(loss_func.type, 'hinge')
  %   mpos = maxb*ncases;
  %   while (npos < mpos)
  %     if (isfield(data, 'nnTraining'))
  % 	x1ndtmp = ceil(rand(mpos,1)*Ntraining2);      
  % 	nnn = size(data.nnTraining,1);
  % 	x2ndtmp = double(data.nnTraining(ceil(rand(mpos,1).*(nns(x1ndtmp)))+(x1ndtmp-1)*nnn));
  % 	casespos = sub2ind([Ntraining2 Ntraining1], x1ndtmp, x2ndtmp);
  % 	npos = mpos;
  %     else
  % 	tmp = ceil(rand(mpos,1)*NtrainingSqr); 
  % 	if (isfield(data, 'Ltraining'))
  % 	  [x1ndtmp x2ndtmp] = ind2sub([Ntraining2 Ntraining1], tmp);
  % 	  tmppos = find(data.Ltraining(x1ndtmp) == data.Ltraining(x2ndtmp));
  % 	else
  % 	  tmppos = find(data.Straining(tmp));
  % 	end
  % 	ntmppos = numel(tmppos);    
  % 	if (npos+ntmppos > mpos)
  % 	  ntmppos = mpos - npos;
  % 	  tmppos = tmppos(1:ntmppos);
  % 	end
  % 	casespos(npos+1:npos+ntmppos) = tmp(tmppos);
  % 	npos = npos+ntmppos;
  %     end
  %   end
  %   mneg = maxb*ncases;
  %   while (nneg < mneg)
  %     tmp = ceil(rand(mneg,1)*NtrainingSqr);
      
  %     [x1ndtmp2 x2ndtmp2] = ind2sub([Ntraining2 Ntraining1], tmp);
  %     if (isfield(data, 'Ltraining'))
  % 	tmpneg = find(data.Ltraining(x1ndtmp2) ~= data.Ltraining(x2ndtmp2));
  %     else
  % 	tmpneg = find(data.Straining(tmp) == 0);
  %     end
  %     ntmpneg = numel(tmpneg);
  %     if (nneg+ntmpneg > mneg)
  % 	ntmpneg = mneg - nneg;
  % 	tmpneg = tmpneg(1:ntmpneg);
  %     end
  %     casesneg(nneg+1:nneg+ntmpneg) = tmp(tmpneg);
  %     nneg = nneg+ntmpneg;    
  %   end
  % end
  
  if (exist('nnTraining2', 'var'))
    % Re-select the nearest neighbors
    fprintf('.');
    if (~iscell(W))
      B1 = W*[Xtraining];
    else
      resp = compute_NN_output(W, [Xtraining], param.nonlinearity);
      B1 = resp{end};
      clear resp
    end
    
    B1 = logical(single(B1 > 0));
    B1 = compactbit(B1);
    for l = data.labels(:)'
      which_l = data.Ltraining == l;
      id_l = find(which_l);
      id_nl = find(~which_l);

      % [b retrieved c d] = linscan_sorted_multi_mex(B1(:,which_l), B1(:,which_l), sum(which_l), nb, ceil(nb/2), data.nn+1, 16);
      % for i=1:numel(id_l)
      %   nnTraining(:,id_l(i)) = id_l(retrieved(2:end, i));
      % end
      [b retrieved c d] = linscan_sorted_multi_mex(B1(:,~which_l), B1(:,which_l), sum(which_l), nb, ceil(nb/2), data.nn2, 6);
      for i=1:numel(id_l)
	nnTraining2(:,id_l(i)) = id_nl(retrieved(:, i));
      end
    end
  end
  
  for b=1:maxb
    % if strcmp(loss_func.type, 'hinge')
    %   % make the fraction of positive pairs to be at least lambda
    %   ncases2 = min(round(ncases * lambda), ncases); 
    %   ncases1 = ncases - ncases2;      
    %   cases(1:ncases1) = casesneg(nneg-ncases1+1:nneg);
    %   nneg = nneg - ncases1;
    %   cases(ncases1+1:end) = casespos(npos-(ncases-ncases1)+1:npos);
    %   npos = npos - (ncases-ncases1);
    %   [x1nd x2nd] = ind2sub([Ntraining2 Ntraining1], cases);
    % elseif strcmp(loss_func.type, 'triplet-ranking') || strcmp(loss_func.type, 'triplet-ranking2')
    %   if (exist('nnTraining2', 'var'))
    % 	x1nd = ceil(rand(ncases,1)*Ntraining1);
    % 	nnn2 = size(nnTraining2,1);
    % 	x2nd = double(nnTraining2(ceil(rand(ncases,1).*(nnn2))+(x1nd-1)*nnn2));
    %   else
    % 	s = ceil(rand(ncases*2, 1)*NtrainingSqr);
    % 	[x1ndt x2ndt] = ind2sub([Ntraining2 Ntraining1], s);
    % 	f = find(data.Ltraining(x1ndt) ~= data.Ltraining(x2ndt));
    % 	x1nd = x1ndt(f(1:ncases));
    % 	x2nd = x2ndt(f(1:ncases));
    %   end
    %   nnn = size(data.nnTraining,1);
    %   x3nd = double(nnTraining(ceil(rand(ncases,1).*(nns(x1nd)))+(x1nd-1)*nnn));
    % end
    
    if strcmp(loss_func.type, 'hinge')
      x1nd                   = sbatch1( (b-1)*(ncases)+1         : (b)*(ncases) );
      x2nd(1:ncases1)        = sbatch2( (b-1)*(ncases)+1         : (b-1)*(ncases)+ncases1 );
      x2nd(ncases1+1:ncases) = sbatch3( (b-1)*(ncases)+ncases1+1 : (b)*(ncases) );
    elseif strcmp(loss_func.type, 'triplet-ranking') || strcmp(loss_func.type, 'triplet-ranking2')
      x1nd = sbatch1( (b-1)*(ncases)+1 : (b)*(ncases) );
      x2nd = sbatch2( (b-1)*(ncases)+1 : (b)*(ncases) );
      x3nd = sbatch3( (b-1)*(ncases)+1 : (b)*(ncases) );
    end
    
    if (isa(Xtraining, 'uint8'))
      x1 = bsxfun(@minus, double(Xtraining(:, x1nd(:))), data.learn_mean) * data.scale;
    else
      x1 = (Xtraining(:, x1nd(:)));
      if (param.noisy)
	x1 = x1 + grandn(input_dim, ncases)*.3;
      end
    end
    
    % if (param.gpu == 1)
    %   x1 = gdouble(x1);
    % end    

    if (~iscell(W))
      Wx1 = W*x1;
    else
      % resp1 = compute_NN_output(W, x1);
      % Wx1 = resp1{end} - .5;
      resp1 = compute_NN_output(W, x1, param.nonlinearity);
      Wx1 = resp1{end};
    end
    y1 = sign(Wx1);
    % y1 is zero where Wx1 is zero
    
    if (isa(Xtraining, 'uint8'))
      x2 = bsxfun(@minus, double(Xtraining(:, x2nd(:))), data.learn_mean) * data.scale;
    else
      x2 = (Xtraining(:, x2nd(:)));
      if (param.noisy)
	x2 = x2 + grandn(input_dim, ncases)*.3;
      end
    end
    
    % if (param.gpu == 1)
    %   x2 = gdouble(x2);
    % end

    if (~iscell(W))
      Wx2 = W*x2;
    else
      % resp2 = compute_NN_output(W, x2);
      % Wx2 = resp2{end} - .5;
      resp2 = compute_NN_output(W, x2, param.nonlinearity);
      Wx2 = resp2{end};
    end
    y2 = sign(Wx2);			% we use -1/+1 instead of 0/1 values for the binary vectors
    % y2 is zero where Wx2 is zero
        
    if strcmp(loss_func.type, 'triplet-ranking') || strcmp(loss_func.type, 'triplet-ranking2')
      % hdis2d = single(nb-(nb/2+((y1'*y2)/2)));
      hdis2d = nb-(nb/2+((y1'*y2)/2));
      samelabel = bsxfun(@eq, data.Ltraining(x1nd(:)), data.Ltraining(x2nd)');
      hdis2d(samelabel) = nb*2;
      [h ih] = sort(hdis2d, 2);
      closest_ih = ih(:,1);    
      x2nd = x2nd(closest_ih);
      x2 = x2(:,closest_ih);
      y2 = y2(:,closest_ih);
      Wx2 = Wx2(:,closest_ih);
      if (iscell(W))
      	for i=1:numel(W)
      	  resp2{i} = resp2{i}(:,closest_ih);
      	end
      end

      if (isa(Xtraining, 'uint8'))
	x3 = bsxfun(@minus, double(Xtraining(:, x3nd(:))), data.learn_mean) * data.scale;
      else
	x3 = Xtraining(:, x3nd(:));
	if (param.noisy)
	  x3 = x3 + grandn(input_dim, ncases)*.3;
	end
      end
      
      % if (param.gpu == 1)
      % 	x3 = gdouble(x3);
      % end
      
      if (~iscell(W))
	Wx3 = W*x3;
      else
	% resp3 = compute_NN_output(W, x3);
	% Wx3 = resp3{end} - .5;
	resp3 = compute_NN_output(W, x3, param.nonlinearity);
	Wx3 = resp3{end};
      end
      y3 = sign(Wx3);			% we use -1/+1 instead of 0/1 values for the binary vectors
    end

    if (param.gpu == 2)
      GPUzeros(y1p);
      GPUzeros(y2p);
      GPUzeros(y3p);
    end
    
    % sanity check of the labels
    if (b == 1)
      if (strcmp(loss_func.type, 'triplet-ranking') || strcmp(loss_func.type, 'triplet-ranking2'))
	if (any(data.Ltraining(x1nd) == data.Ltraining(x2nd)) || ...
	    any(data.Ltraining(x1nd) ~= data.Ltraining(x3nd)))
	  [data.Ltraining(x1nd) data.Ltraining(x2nd) data.Ltraining(x3nd)]
	  error('Checking triplet labels... something is wrong!');
	end
      elseif (strcmp(loss_func.type, 'hinge'))
	% whos x1nd x2nd l
	
	if any(l(:) ~= data.Straining(sub2ind([Ntraining2 Ntraining1], x1nd, x2nd)))
	  error('Checking pairwise labels... something is wrong!');
	end
      end
    end
    
    if (strcmp(loss_func.type, 'hinge'))
      [y1p y2p nflip] = loss_aug_inf_mex0(double(Wx1), double(Wx2), double(loss));
    elseif (strcmp(loss_func.type, 'bre'))
      l = full(data.Straining(cases)');
      % creating the quadratic BRE loss function
      % it requires the Dtraining matrix
      d = data.Dtraining(cases);
      loss = ((repmat(d, [1 nb+1]) - repmat((0:nb)/nb, [ncases 1])).^2)';
    
      [y1p y2p nflip] = loss_aug_inf_mex0(double(Wx1), double(Wx2), double(loss));
    elseif (strcmp(loss_func.type, 'triplet-ranking') || strcmp(loss_func.type, 'triplet-ranking2'))
      [y1p y3p y2p ndiff wxloss] = loss_aug_inf_triplet_mex0(double(Wx1), double(Wx3), double(Wx2), double(loss));
      if (param.gpu == 2)
    	y1p = GPUsingle(y1p);
    	y2p = GPUsingle(y2p);
    	y3p = GPUsingle(y3p);
      end
    else
      error('losstype is not supported.\n');
    end

    % sanity check
    if ~param.regularizer && (~param.fast || b == 1)
      if (strcmp(loss_func.type, 'hinge') || strcmp(loss_func.type, 'bre'))
	diffr = (sum(y1.*Wx1+y2.*Wx2)   + loss((0:(ncases-1))*(nb+1)+(sum(y1 ~= y2)+1))) - ...
		(sum(y1p.*Wx1+y2p.*Wx2) + loss((0:(ncases-1))*(nb+1)+(sum(y1p ~= y2p)+1)));
	if any(diffr > 1e-2)
	  display(diffr);
	  error('something is wrong');
	end
      elseif (strcmp(loss_func.type, 'triplet-ranking') || strcmp(loss_func.type, 'triplet-ranking2'))
	diffr = (sum(y1.*Wx1 + y2.*Wx2 + y3.*Wx3) + loss((0:(ncases-1))*(2*nb+1)+(nb+1+sum(y1~=y3)-sum(y1~=y2)))) - ...
		(sum(y1p.*Wx1 + y2p.*Wx2 + y3p.*Wx3) + loss((0:(ncases-1))*(2*nb+1)+(nb+1+sum(y1p~=y3p)-sum(y1p~=y2p))));
	if any(diffr > 1e-2)
	  display(diffr);
	  error('something is wrong');
	end
      end
    end    
    
    meanWx1 = (mean(Wx1,2)-0);
    meanWx1_trunc = meanWx1;
    meanWx1_trunc(meanWx1_trunc > .5) = .5;
    meanWx1_trunc(meanWx1_trunc < -.5) = -.5;

    if (param.regularizer)
      y1minusy1p = bsxfun(@plus, y1 - y1p, -param.regularizer*meanWx1_trunc);
      y2minusy2p = y2-y2p;
      % y1minusy1p = bsxfun(@plus, zeros(size(y1)), -param.regularizer*(mean(Wx1,2)-0));
      % y2minusy2p = zeros(size(y2));
    else
      y1minusy1p = y1-y1p;
      y2minusy2p = y2-y2p;
    end
    nonzero_grad_1 = sum(y1 ~= y1p) ~= 0;
    nonzero_grad_2 = sum(y2 ~= y2p) ~= 0;
    if (strcmp(loss_func.type, 'triplet-ranking') || strcmp(loss_func.type, 'triplet-ranking2'))
      y3minusy3p = y3-y3p;
      nonzero_grad_3 = sum(y3 ~= y3p) ~= 0;
    end
    
    % gradient
    if (~iscell(W))
      % if (param.gpu)
      	grad = (y1minusy1p * x1' + y2minusy2p * x2');
      % else
      % 	grad = (y1minusy1p(:,nonzero_grad_1) * x1(:,nonzero_grad_1)' + ...
      % 		y2minusy2p(:,nonzero_grad_2) * x2(:,nonzero_grad_2)');
      % end
      if (strcmp(loss_func.type, 'triplet-ranking') || strcmp(loss_func.type, 'triplet-ranking2'))
	if (param.gpu)
	  grad = grad + (y3minusy3p * x3');
	else
	  grad = grad + (y3minusy3p(:,nonzero_grad_3) * x3(:,nonzero_grad_3)');
	end
      end
    else
      if (param.gpu)
      	grad1 = compute_NN_grad(W, x1, resp1, y1minusy1p, param.nonlinearity);
      	grad2 = compute_NN_grad(W, x2, resp2, y2minusy2p, param.nonlinearity);
      else
      	grad1 = compute_NN_grad(W, x1, resp1, y1minusy1p, param.nonlinearity);
	% grad1 = compute_NN_grad(W, x1, resp1, y1minusy1p, param.nonlinearity, nonzero_grad_1);
	grad2 = compute_NN_grad(W, x2, resp2, y2minusy2p, param.nonlinearity, nonzero_grad_2);
      end
      if (strcmp(loss_func.type, 'triplet-ranking') || strcmp(loss_func.type, 'triplet-ranking2'))
	if (param.gpu)
	  grad3 = compute_NN_grad(W, x3, resp3, y3minusy3p, param.nonlinearity);
	else
	  grad3 = compute_NN_grad(W, x3, resp3, y3minusy3p, param.nonlinearity, nonzero_grad_3);
	end
      end
    end

    if (verbose) % debug information
      n_bits_on    = n_bits_on    + sum(y1==1, 2) + sum(y2==1, 2);
      n_bits_total = n_bits_total + 2*ncases;
      
      r = n_bits_on / n_bits_total;
      bits_useless   = (min(r, 1-r)) < .02;
      n_bits_useless = sum(bits_useless);
      if (n_bits_useless > nb/2)
	if (~do_break)
	  fprintf('** epoch: %d, iter: %d\n', t, b);
	  fprintf('** More than half of the bits have become useless!\n');
	end
	do_break = 1;
	break;
      end      
      
      hdis = sum(y1 ~= y2);
      if (strcmp(loss_func.type, 'triplet-ranking') || strcmp(loss_func.type, 'triplet-ranking2'))
	hdis  = nb + sum(y1 ~= y3) - hdis;	
	hdis2 = nb + sum(y1p ~= y3p) - sum(y1p ~= y2p);
	nonzero = nonzero_grad_1 | nonzero_grad_2 | nonzero_grad_3;
	nnz = nnz + sum(nonzero);
	
	if (b == 1)
	  if (hdis2(:) ~= (nb + ndiff(:)))
	    
	    Wx1 = double(Wx1);
	    Wx2 = double(Wx3);
	    Wx3 = double(Wx2);
	    loss = double(loss);
	    
	    save ~/wrong Wx1 Wx2 Wx3 loss;

	    hdis(1:10)
	    ndiff(1:10)'
	    nb + ndiff(1:10)
	    whos ndiff hdis
	    error('hids ~= ndiff');
	  end
	end
	
	fracs = fracs + [sum((hdis>=nb)&nonzero), sum(hdis>=nb)];
      else
	nonzero = nonzero_grad_1 | nonzero_grad_2;
	nnz = nnz + sum(nonzero);

	FP_FN = (hdis>=rho & l==1) | (hdis<=rho & l==0);
	fracs = fracs + [sum(FP_FN & nonzero), sum(FP_FN)];
      end
      
      if (~iscell(W))
	normW = .5 * shrink_w * sum(sum(W.^2));
      else
	normW = 0;
	for i=1:numel(W)
	  normW = normW + .5 * shrink_w(i) * sum(sum(W{i}.^2));
	end
      end
      
      if (strcmp(loss_func.type, 'triplet-ranking') || strcmp(loss_func.type, 'triplet-ranking2'))
	if (param.gpu == 2)
	  hdis2 = single(hdis2);
	  hdis = single(hdis);
	end
	
	if (~iscell(W))
	  batch_bound = ncases * normW - sum(sum(W .* grad)) + sum(loss(hdis2+1+(0:ncases-1)*(2*nb+1))) ...
	      + .5*param.regularizer*sum((meanWx1).^2);
	else
	  batch_bound = ncases * normW - sum(sum(Wx1 .* y1minusy1p)) - sum(sum(Wx2 .* y2minusy2p)) ...
	      - sum(sum(Wx3 .* y3minusy3p)) + sum(loss(hdis2+1+(0:ncases-1)*(2*nb+1))) + .5*param.regularizer*sum((meanWx1).^2);
	end
      else
	if (~iscell(W))
	  batch_bound = ncases * normW - (sum(sum(W .* grad)) + sum(loss(nflip+1+(0:ncases-1)*(nb+1)))) ...
	      + .5*param.regularizer*sum((meanWx1).^2);
	else
	  batch_bound = ncases * normW - sum(sum(Wx1 .* y1minusy1p)) - sum(sum(Wx2 .* y2minusy2p)) ...
	      + sum(loss(nflip+1+(0:ncases-1)*(nb+1))) + .5*param.regularizer*sum((meanWx1).^2);
	end
      end
      bound = bound + batch_bound;
      
      if (strcmp(loss_func.type, 'triplet-ranking') || strcmp(loss_func.type, 'triplet-ranking2'))
	batch_emploss = (ncases * normW) + .5*param.regularizer*sum((meanWx1).^2) + sum(loss((0:(ncases-1))*(2*nb+1)+(hdis+1)));
      else
	batch_emploss = (ncases * normW) + .5*param.regularizer*sum((meanWx1).^2) + sum(loss((0:(ncases-1))*(nb+1)+(hdis+1)));
      end	
      emploss = emploss + batch_emploss;
    end
    
    
    % % sanity check: to make sure W and Winc are double-precision throughout the learning
    % if (b == 1)
    %   if (param.gpu == 1)
    % 	if ~iscell(W) && (~isa(W, 'gdouble') || ~isa(Winc, 'gdouble'))
    % 	  error('W or Winc is not a gdouble');
    % 	end
    %   elseif (param.gpu == 0)
    % 	if ~iscell(W) && (~isa(W, 'double') || ~isa(Winc, 'double'))
    % 	  error('W or Winc is not a double')
    % 	end
    %   end
    % end
    
    % if (b == 1 && t == 2)
    %   whos grad ncases shrink_w W
    %   d1 = double(grad ./ ncases - shrink_w * W);
    %   d2 = double(grad) ./ ncases - shrink_w * W;
    %   d1(1:10)
    %   d2(1:10)
    % end
    
    % update rule of W
    if (param.gpu == 0)
      if (~iscell(W))
	if (zerobias)
	  Winc = momentum * Winc ...
		 + eta * double(grad)/ncases ...
		 - eta * shrink_w * W;
	  Winc(:,end) = 0;
	else
	  Winc = momentum * Winc ...
		 + eta * double(grad)/ncases ...
		 - eta * shrink_w * W;
	end
	W = W + Winc;
      else
	for i=1:numel(W)
	  if (strcmp(loss_func.type, 'triplet-ranking') || strcmp(loss_func.type, 'triplet-ranking2'))
	    Winc{i} = momentum * Winc{i} ...
		      + eta * double(grad1{i} + grad2{i} + grad3{i})/ncases ...
		      - eta * shrink_w(i) * W{i};
	  else
	    Winc{i} = momentum * Winc{i} ...
		      + eta * double(grad1{i} + grad2{i})/ncases ...
		      - eta * shrink_w(i) * W{i};
	  end
	  W{i} = W{i} + Winc{i};
	end
      end
    elseif (param.gpu == 1)
      if (~iscell(W))
	if (zerobias)
	  Winc = momentum * Winc ...
		 + eta * grad/ncases ...
		 - eta * shrink_w * W;
	  Winc(:,end) = 0;
	else
	  Winc = momentum * Winc ...
		 + eta * grad/ncases ...
		 - eta * shrink_w * W;
	end
	W = W + Winc;
      else
	for i=1:numel(W)
	  if (strcmp(loss_func.type, 'triplet-ranking') || strcmp(loss_func.type, 'triplet-ranking2'))
	    Winc{i} = momentum * Winc{i} ...
		      + eta * (grad1{i} + grad2{i} + grad3{i})/ncases ...
		      - eta * shrink_w(i) * W{i};
	  else
	    Winc{i} = momentum * Winc{i} ...
		      + eta * (grad1{i} + grad2{i})/ncases ...
		      - eta * shrink_w(i) * W{i};
	  end
	  W{i} = W{i} + Winc{i};
	end
      end
    end
    
    
    % % we don't re-normalize rows of W as mentioned in the paper anymore, instead we use weight decay
    % % i.e., L2 norm regularizer    
    % if (~iscell(W) && shrink_w == 0)
    %   % normW2 = sqrt(sum(W(:,1:end-1).^2,2));
    %   normW2 = sqrt(sum(W.^2,2));
    %   normW2(normW2 < 1) = 1;
    %   W = bsxfun(@rdivide, W, normW2);
    % end
  end
  time_iter = toc(tic_id_iter);

  if do_break
    break;
  end
  
  fprintf('(%3d/%.1d/%.2fs/%.2f)', t, eta, time_iter, double(sum(meanWx1.^2)));
  
  if (~verbose || (~param.nval_during && ~param.nval_after) || (verbose && mod(t, verbose) ~= 0))
    fprintf('\r');
  end

  if (verbose && t <= maxiter && mod(t, verbose) == 0)    
    if (~iscell(W))
      normW = .5 * sum(W(:).^2);
    else
      normW = 0;
      for i=1:numel(W)
	normW = normW + .5 * sum(sum(W{i}.^2));
      end
    end
    
    time_val = toc(time_val_tic);
    
    rr = sort(min(r(:), 1 - r(:)))';
    fprintf([' ~~~ bound:%.3f > loss:%.3f ~~~ norm W=%.2f ~~~ ' ...
	     ' act:%.3f %.3f %.3f ~~~ #useless bits:%d ~~~ #nzg:%.2f ~~~ fracs:%.2f %.2f ~~~ t:%.2f'], double(bound/(loss_func.scale*maxb*verbose)), ...
	    double(emploss/(loss_func.scale*maxb*verbose)), double(normW), double(rr(1:3)), double(n_bits_useless), ...
	    double(nnz/(maxb*verbose)), double(fracs(1)/(maxb*verbose)), double(fracs(2)/(maxb*verbose)), time_val/verbose);
    fprintf('\n');

    param.bound(t/verbose) = double(bound/(loss_func.scale*maxb*verbose));
    param.emploss(t/verbose) = double(emploss/(loss_func.scale*maxb*verbose));
    param.etas(t/verbose) = eta;

    % Bold Driver learning rate adaptation
    if (shrink_eta == 2 && t >= 3*verbose)
      if (last_bound < bound && last_emploss < emploss) % last2_bound < bound)
      	eta = .5 * eta;
      	if (eta < param.mineta)
      	  eta = param.mineta;
      	end
      elseif (last_bound > bound && last_emploss > emploss) % && last2_bound > last_bound)
      	eta = 1.05 * eta;
      	if (eta > param.maxeta)
      	  eta = param.maxeta;
      	end
      end
    end
    % last2_bound = last_bound;
    last_bound = min(last_bound, bound);
    % last2_emploss = last_emploss;
    last_emploss = min(emploss, last_emploss);
    
    time_val_tic = tic;
    bound = 0;
    emploss = 0;
    nnz = 0;
    fracs = [0 0];
  
    Wall{t+1} = W;
  
    % Rehab of bits :)
    if (n_bits_useless > 0)
      n_bits_useless = double(n_bits_useless);
      if ~iscell(W)
	W(bits_useless, :) = .01*randn(n_bits_useless, size(W,2));
	W(bits_useless,end) = W(bits_useless,end) - median(W(bits_useless,:) * data.Xtraining, 2);
	% W(bits_useless, end) = 0;
      else
	W{end}(bits_useless, :) = .01 * randn(n_bits_useless, size(W{end},2));
	resp_tmp = compute_NN_output(W, data.Xtraining, param.nonlinearity);
	W{end}(bits_useless,end) = -median(W{end}(bits_useless,1:end-1)*(resp_tmp{numel(W)-1}),2);
      end
      fprintf('(+%d)\n', n_bits_useless);
    end
  end
  
  if (param.nval_during && t < maxiter)
    if (mod(t, tval) == 0)
      [ap acc acc2] = eval(data, W, param, rho, verbose);
      param.testacc(ceil(t/tval)) = acc;
      param.trainacc(ceil(t/tval)) = acc2;      
    end
  end
  
  if (param.nval_after && t >= maxiter)
    [ap acc acc2] = eval(data, W, param, rho, verbose);
    param.testacc(floor((maxiter-1)/tval)+t-maxiter+1) = acc;
    param.trainacc(floor((maxiter-1)/tval)+t-maxiter+1) = acc2;      
    if (isinf(avg_acc))
      avg_acc = 0;
    end
    avg_acc = avg_acc + acc;
    mean_ap = mean_ap  + ap;
  end
end

time_full = toc(tic_id_full);

if (param.nval_after)
  mean_ap  = mean_ap  / param.nval_after;
  avg_acc = avg_acc / param.nval_after;
  if (verbose)
    fprintf('Mean ap over %d final step(s): %.3f', param.nval_after, mean_ap);
    if (~isinf(avg_acc))
      fprintf(' ~~~ mean accuracy: %.4f', avg_acc);
    end
    fprintf(' ~~~ time(s): %.2f', time_full)
    fprintf('                   \n');
  end
end

param.x_dim = input_dim;
param.nns = data.nns;
param.ap  = mean_ap;
param.acc = avg_acc;
param.l = loss(:,1);
final_W = W;
final_params = param;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ap acc acc2] = eval(data, W, param, rho, verbose)

acc = NaN;  % acc and acc2 can be used to return any other types of accuracy concerned.
acc2 = NaN;

if (strcmp(data.MODE(1:min(numel(data.MODE),9)), 'sem-mnist') || strcmp(data.MODE(1:min(numel(data.MODE),8)), 'cifar-10')) %mnist
  knn = 3;
  [acc acc2] = eval_hammknn(data, W, knn, param.nonlinearity);
  acc = acc(knn);
  acc2 = acc2(knn);
end

if ~(strcmp(data.MODE, 'inria-sift-1B') || strcmp(data.MODE, 'inria-sift-1M') || strcmp(data.MODE(1:min(numel(data.MODE),9)), 'sem-mnist') || strcmp(data.MODE(1:min(numel(data.MODE),8)), 'cifar-10'))
  [p1 r1] = eval_linear_hash(W, data);
  p1(isnan(p1)) = 1;
end

ap = 0;
if (strcmp(data.MODE, 'sem-22K-labelme'))
  % semantic labelme
  [ap pcode] = eval_labelme(W, data);
  if (verbose)
    fprintf(['prec(rho<=%d): %.3f ~~~ recall(rho<=%d): %.4f ~~~ ap: %.5f ~~~ ap(50): %.3f' ...
	     ' ~~~ ap(100): %.3f ~~~ ap(500): %.3f'], rho, p1(rho+1), rho, r1(rho+1), ...
	    ap, pcode(50), pcode(100), pcode(500));
  end
elseif (strcmp(data.MODE, 'inria-sift-1B') | strcmp(data.MODE, 'inria-sift-1M'))
  [rec rec2] = eval_nn(W, data);
  ap = mean(rec);
  ap2 = mean(rec2);
  if (verbose)
    % fprintf(['%s set: prec(rho<=%d): %.3f ~~~ recall(rho<=%d): %.4f ~~~ rec@1: %.5f ~~~ rec@10: %.3f' ...    rho, p1(rho+1), rho, r1(rho+1)
    fprintf(['%s set: ap: %.3f ~~~ rec@1: %.3f ~~~ rec@10: %.3f ~~~ rec@100: %.3f ~~~ rec@1000: %.3f ~~~ |' ...
	     ' ap: %.3f ~~~ rec@1: %.3f ~~~ rec@10: %.3f ~~~ rec@100: %.3f ~~~ rec@1000: %.3f\n'], ...
	     ap, rec(1), rec(10), rec(100), rec(1000), ap2, rec2(1), rec2(10), rec2(100), rec2(1000));
  end
elseif (~strcmp(data.MODE(1:min(numel(data.MODE),9)), 'sem-mnist') && ~strcmp(data.MODE(1:min(numel(data.MODE),8)), 'cifar-10')) %mnist
  ap = sum([(p1(1:end-1)+p1(2:end))/2].*[(r1(2:end)-r1(1:end-1))]);
  if (verbose)
    fprintf('prec(rho<=%d): %.3f ~~~ recall(rho<=%d): %.4f ~~~ ap: %.5f', rho, ...
	    p1(rho+1), rho, r1(rho+1), ap);
  end
end

if (~isnan(acc))
  fprintf(' ~~~ acc(test): %.4f ~~~ acc(training): %.4f ', acc, acc2);
end

fprintf('\n');
