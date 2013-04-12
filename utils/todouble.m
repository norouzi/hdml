function W = todouble(W)

for j = 1:numel(W)
  if (iscell(W(j).W))
    for k=1:numel(W(j).W)
      W(j).W{k} = double(W(j).W{k});
    end
  else
    W(j).W = double(W(j).W);
  end
  W(j).params.l = double(W(j).params.l);
end
