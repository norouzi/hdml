function [neighbors nns] = sameclass_nn(data, nn, W)

if (exist('W', 'var'))
  if (~iscell(W))
    B1 = W * data.Xtraining;
  else
    resp1 = compute_NN_output3(W, data.Xtraining);
    B1 = resp1{end};
  end

  B1 = logical(single(B1 > 0));
  B1 = compactbit(B1);
  [nw1 ~] = size(B1);
  nb = nw1*8;
end


nl = zeros(numel(data.labels(:)), 1);
for l = data.labels(:)'
  nl(l+1) = sum(data.Ltraining == l);
end

nns = zeros(data.Ntraining,1);
neighbors = zeros(min(nn, min(nl)-1), data.Ntraining, 'single');

for l = data.labels(:)'
  which_l = data.Ltraining == l;
  id_l = find(which_l);
  nn_l = min(nn, nl(l+1)-1);
  nns(which_l) = nn_l;
  if (~exist('W', 'var'))
    [nnTraining_l ~] = yael_nn(single(data.Xtraining(:,which_l)), ...
			       single(data.Xtraining(:,which_l)), nn_l+1 );
  else
    [btmp nnTraining_l ctmp dtmp] = linscan_sorted_multi_mex(B1(:,which_l), B1(:,which_l), nl(l+1), nb, ceil(nb/2), nn_l+1, 4);
  end
  
  for i=1:numel(id_l)
    neighbors(1:nn_l,id_l(i)) = id_l(nnTraining_l(2:end, i));
  end
end
