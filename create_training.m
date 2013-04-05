function data = create_training(data, trainset, doval)

if (strcmp(trainset, 'train'))
  Ntraining = data.Ntraining;
  Xtraining = data.Xtraining;

  % one tenth of the training points are used for validation
  Ntest = min(ceil(Ntraining / 5), 10000);
  Ntraining = Ntraining - Ntest;
  
  % we re-define test set to be the validation set.
  % this way all the evaluation codes remain unchanged.
  Xtest = Xtraining(:, Ntraining+1:end);
  % StestTraining(StestTraining  == -1) = 0;
  if (isfield(data, 'StrainingNeg'))
    StestTrainingNeg = data.StrainingNeg(Ntraining+1:end, 1:Ntraining);
  end

  Xtraining = Xtraining(:, 1:Ntraining);
  if (isfield(data, 'Straining'))
    Straining = data.Straining;
    StestTraining = Straining(Ntraining+1:end, 1:Ntraining);
    Straining = Straining(1:Ntraining, 1:Ntraining);
  end
  if (isfield(data, 'StrainingNeg'))
    StrainingNeg = data.StrainingNeg(1:Ntraining, 1:Ntraining);
  end
  if (isfield(data, 'Dtraining'))
    Dtraining = data.Dtraining(1:Ntraining, 1:Ntraining);  
    DtestTraining = data.Dtraining(Ntraining+1:end, 1:Ntraining);
  end

  % if some kind of labeling exists e.g., class labels
  if (isfield(data, 'Ltraining'))
    Ltest = data.Ltraining(Ntraining+1:end,:);
    Ltraining = data.Ltraining(1:Ntraining,:);
  end

  if (isfield(data, 'nnTraining'))
    data.nnTest = data.nnTraining(:, Ntraining+1:end);
    data.nnTraining = data.nnTraining(:, 1:Ntraining);
  end
  data.Xtraining = Xtraining;
  if (isfield(data, 'Straining'))
    data.Straining = Straining;
    data.StestTraining = StestTraining;
  end
  data.Ntraining = Ntraining;
  if (isfield(data, 'StrainingNeg'))
    data.StrainingNeg = StrainingNeg;
    data.StestTrainingNeg = StestTrainingNeg;
  end
  if (isfield(data, 'Dtraining'))
    data.Dtraining = Dtraining;
  end
  if (isfield(data, 'Ltraining'))
    data.Ltest = Ltest;
    data.Ltraining = Ltraining;
  end
  if (exist('Xtest') && doval)
    if (isfield(data, 'DtestTraining'))
      data.DtestTraining = DtestTraining;
    end
    data.Xtest = Xtest;  
    data.Ntest = Ntest;
  end
  
  if (isfield(data, 'indPos') && isfield(data, 'Straining'))
    data.indPos = find(data.Straining(:));
  end

elseif (strcmp(trainset, 'trainval'))
  return;
  % ordinary train and test sets are used
  if (isfield(data, 'Dtraining'))
    Dtraining = data.Dtraining;
  end
  if (isfield(data, 'DtestTraining'))
    DtestTraining = data.DtestTraining;
  end
  if (isfield(data, 'Xtest'))
    Ntest = data.Ntest;
    Xtest = data.Xtest;
    StestTraining = data.StestTraining;

    % if some kind of labeling exists e.g., class labels    
    if (isfield(data, 'Ltraining'))
      Ltest = data.Ltest;
      Ltraining = data.Ltraining;
    end
  end
else
  error('trainset should be either "trainval" or "train"');
end

if (isfield(data, 'nnlabels'))
  data.nnlabels = zeros(numel(data.labels),1);
  for i = 1:numel(data.labels(:))
    data.nnlabels(i) = sum(data.Ltraining(:) ~= data.labels(i));
  end  
  data.indnlabels = zeros(max(data.nnlabels), numel(data.labels));
  for i = 1:numel(data.labels(:))
    data.indnlabels(1:data.nnlabels(i),i) = find(data.Ltraining(:) ~= data.labels(i));
  end
end

if (~doval)
  if (isfield(data, 'Xtest'))
    data = rmfield(data, 'Xtest');
    data = rmfield(data, 'Ntest');
  end
  if (isfield(data, 'DtestTraining'))
    data = rmfield(data, 'DtestTraining');
  end
  if (isfield(data, 'StestTraining'))
    data = rmfield(data, 'StestTraining');
  end
end
