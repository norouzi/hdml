gpu = GPU;
which_loss = LOSS;

fprintf('DATA = ''%s''; GPU = %d; LOSS = %d; NN = %.0f; NB = [', DATA, GPU, LOSS, NN);
for i=1:numel(NB)-1
  fprintf('%d ', NB(i));
end
fprintf('%d];\n', NB(end));

options = {'FINAL', 'REG', 'SBATCH', 'SHRINK_W', 'ETA', 'MOMENTUM', 'ZERO', 'MARGIN', ...
	   'LSCALE', 'RHO'};
for i=1:numel(options)
  if (exist(options{i}, 'var'))
    fprintf('%s = %d\n', options{i}, eval(options{i}));
  end
end

if (~exist('FINAL', 'var'))
  FINAL = 0;
end

if which_loss == 0
  loss_type = 'hinge';
elseif which_loss == 1
  loss_type = 'triplet-ranking';
else  
  error('Loss is not supported');
end
fprintf('(%s)\n', loss_type);

if (strcmp(DATA, 'mnist'))
  DATASET = 'sem-mnist'
  if (LOSS < 10)
    operand = 0;
  else
    operand = 1;
  end
elseif (strcmp(DATA, 'cifar10'))
  DATASET = 'cifar-10'
  operand = 5;
end

if (FINAL)
  folder_name = sprintf('res/FINAL/NN_%d_L%d', NN, LOSS);
else
  folder_name = sprintf('res/NN_%d_L%d', NN, LOSS);
end

file_name = ['W_', DATA, '_', strrep(num2str(NB), '  ', '_')];
if (~exist(folder_name, 'file'))
  eval(['! mkdir -p ', folder_name]);
end

folder_name
file_name

clear data2;
if (gpu == 1)
  data2 = create_data_hdml([DATASET, '-gpu'], operand);
elseif (gpu == 2)
  data2 = create_data_hdml([DATASET, '-GPU'], operand);
else
  data2 = create_data_hdml(DATASET, operand);
end
if (~FINAL)
  data2 = create_training(data2, 'train', 1);
end
[data2.nnTraining data2.nns] = sameclass_nn(data2, Inf);
data2.nn = size(data2.nnTraining,1);
if (~isinf(NN))
  data2.nns = ones(size(data2.nns)) * NN;
end
data2 = addone(data2);

p.nb = NB;
rho = round(p.nb(end)*.35);
if (exist('SBATCH', 'var'))
  p.sbatches = SBATCH;
else
  p.sbatches = 100;
end;
if (exist('ETA', 'var'))
  p.eta = ETA;
else
  p.eta = 1e-3;
end
p.niter = 2000;
p.etabound = [1e-5; Inf];
p.verbose = 25;
p.nverbose = ceil(.2*p.niter/p.verbose);
if (exist('ZERO', 'var'))
  p.zerobias = ZERO;
else
  p.zerobias = 0;
end
if (exist('MOMENTUM', 'var'))
  p.momentum = MOMENTUM;
else
  p.momentum = .9;
end
p.val_after = 4;
p.shrink_eta = 2;
p.W = [];
initW = [];
p.regularizer = 1;

if (strcmp(DATA, 'mnist'))
  if (which_loss == 0)
    if (numel(NB) == 1)
      p.loss = {'hinge', rho, .5, 1};
      p.shrink_w = [1e-2];
      % p.shrink_w = [1e-2];
    else
      p.loss = {'hinge', rho, .5, 1};
      % p.shrink_w = ones(numel(NB),1) * [1e-5 1e-4 1e-3 1e-2];
      p.shrink_w = ones(numel(NB),1) * [1e-4];
    end
  else
    if (numel(NB) == 1)
      p.shrink_w = [3e-3]; % 1e-3 1e-2];
      p.loss = {'triplet-ranking', 1, 1};
    else
      p.shrink_w = ones(numel(NB),1) * [1e-3];% 1e-3 1e-2];
      p.loss = {'triplet-ranking', 1, 1};
    end
  end
elseif (strcmp(DATA, 'cifar10'))
  if (which_loss == 0)
    if (numel(NB) == 1)
      p.loss = {'hinge', rho, .5, 1};
      p.shrink_w = [1e-2];
      % p.shrink_w = [1e-2];
    else
      p.loss = {'hinge', rho, .5, 1};
      % p.shrink_w = ones(numel(NB),1) * [1e-5 1e-4 1e-3 1e-2];
      p.shrink_w = ones(numel(NB),1) * [1e-3];
    end
  else
    if (numel(NB) == 1)
      p.loss = {'triplet-ranking', 1, 1};
      p.shrink_w = [1e-3 5e-4 2e-3];
    else
      p.loss = {'triplet-ranking', 1, 1};
      p.shrink_w = ones(numel(NB),1) * [1e-3];
    end
  end
elseif (strcmp(DATA, 'cifar100'))
  if (which_loss == 0)
    if (numel(NB) == 1)
      p.loss = {'hinge', rho, .5, 1};
      p.shrink_w = [1e-2];
      % p.shrink_w = [1e-2];
    else
      p.loss = {'hinge', rho, .5, 1};
      % p.shrink_w = ones(numel(NB),1) * [1e-5 1e-4 1e-3 1e-2];
      p.shrink_w = ones(numel(NB),1) * [1e-3];
    end
  else
    p.loss = {'triplet-ranking', 1, 1};
    if (numel(NB) == 1)
      p.shrink_w = [1e-3];
    else
      p.shrink_w = ones(numel(NB),1) * [1e-3 1e-4 1e-5];
      p.eta = .001;
    end
  end
end

if LOSS == 0 && exist('RHO', 'var')
  p.loss{2} = RHO;
end


if exist('LSCALE', 'var')
  if (which_loss == 0)
    p.loss{4} = LSCALE;
  else
    p.loss{2} = LSCALE;
  end
end    

if (exist('MARGIN', 'var'))
  p.loss{3} = MARGIN;
end

if (exist('SHRINK_W', 'var'))
  if (size(SHRINK_W,1) == 1)
    p.shrink_w = ones(numel(NB),1) * SHRINK_W;
  else
    p.shrink_w = SHRINK_W;
  end
end

if (FINAL)
  p.niter = 5000;
  p.nverbose = ceil(.2*p.niter/p.verbose);
  
  W_f = MLH(data2, p.loss, p.nb, p.eta, p.momentum, p.sbatches, 'trainval', p.niter, p.zerobias, p.nverbose, p.val_after, p.verbose, p.shrink_w, p.shrink_eta, p.W, p.etabound, p.regularizer);
  W_f = todouble(W_f);
  save([folder_name, '/', file_name], 'W_f', 'initW', 'p');
  continue;
end


if (size(p.shrink_w, 2) == 1)
  W_shrinkw = [];
else
  fprintf('---- Validating on weight decay parameter ----\n');
  fprintf('%.1d ', p.shrink_w(1,:));
  fprintf('\n');
  fprintf('---------------------------------------------\n');
  
  W_shrinkw = MLH(data2, p.loss, p.nb, p.eta, p.momentum, p.sbatches, 'trainval', p.niter, p.zerobias, p.nverbose, p.val_after, p.verbose, p.shrink_w, p.shrink_eta, p.W, p.etabound, p.regularizer);
  W_shrinkw = todouble(W_shrinkw);
  save([folder_name, '/', file_name], 'initW', 'W_shrinkw');

  W = best_err(W_shrinkw, 'shrink_w');
  fprintf('Best weight decay (%d bits) = %.0d\n', p.nb(end), W.params.shrink_w(1));
  p.shrink_w = W.params.shrink_w;
end

if (LOSS == 0)
  p.loss{2} = mean(p.loss{2}) + ceil(p.nb(end)/16)*[-1 0 +1];
  % if (strcmp(DATA, 'mnist') && numel(NB) == 1 && NB < 128)
  %   p.loss{2} = mean(p.loss{2}) + ceil(p.nb(end)/16)*[-1];
  % end
  if exist('RHO', 'var')
    p.loss{2} = RHO;
  end
end

if (LOSS == 0 && numel(p.loss{2}) > 1)
  fprintf('------------- Validating on rho -------------\n');
  fprintf('%.0d ', p.loss{2});
  fprintf('\n');
  fprintf('---------------------------------------------\n');
  
  W_rho = MLH(data2, p.loss, p.nb, p.eta, p.momentum, p.sbatches, 'trainval', p.niter, p.zerobias, p.nverbose, p.val_after, p.verbose, p.shrink_w, p.shrink_eta, p.W, p.etabound, p.regularizer);
  W_rho = todouble(W_rho);
  save([folder_name, '/', file_name], 'initW', 'W_rho', 'W_shrinkw');
  
  W = best_err(W_rho, 'loss.rho');
  fprintf('Best loss rho (%d bits) = %.0f\n', p.nb(end), W.params.loss.rho);
  p.loss{2} = W.params.loss.rho;
else
  W_rho = NaN;
end

% clear data2
% if (gpu)
%   data2 = create_data_hdml('sem-full-mnist3-gpu');
% else
%   data2 = create_data_hdml('sem-full-mnist3');
% end
% [data2.nnTraining data2.nns] = sameclass_nn(data2, Inf);
% data2.nn = size(data2.nnTraining,1);
% data2.nns = ones(size(data2.nns)) * 1000;
% data2 = addone(data2);

p.niter = 2000;
p.nverbose = ceil(.2*p.niter/p.verbose);

W_final = MLH(data2, p.loss, p.nb, p.eta, p.momentum, p.sbatches, 'trainval', p.niter, p.zerobias, p.nverbose, p.val_after, p.verbose, p.shrink_w, p.shrink_eta, p.W, p.etabound, p.regularizer);
W_final = todouble(W_final);
save([folder_name, '/', file_name], 'initW', 'W_final', 'W_rho', 'W_shrinkw', 'p');

continue;

NB = 64;
for NN = [500 1000 3000 Inf]
  RUN_mnist
end


NB = [64 128];
RUN_mnist;


NB = [64 64 64];
RUN_mnist;


17; DATA = 'mnist'; REG = 1; NB = 32; LOSS = 1; LSCALE = 1; NN = 1000; GPU = 0; SHRINK_W = 3e-3; RUN2;
3;  DATA = 'mnist'; REG = 1; NB = 32; LOSS = 1; LSCALE = 1; NN = 1000; GPU = 0; SHRINK_W = 1e-3; RUN2;
4;  DATA = 'mnist'; REG = 1; NB = 32; LOSS = 1; LSCALE = 1; NN = 1000; GPU = 0; SHRINK_W = 1e-4; RUN2;

% GPU = 0; LOSS = 1; NN = Inf; NB = [1000 64]; DATA = 'cifar10';



W_final2 = MLH(data2, p.loss, p.nb, W_final.params.etas(end), p.momentum, p.sbatches, 'trainval', p.niter, p.zerobias, p.nverbose, p.val_after, p.verbose, p.shrink_w, p.shrink_eta, W_final.W, p.etabound,p.regularizer);

