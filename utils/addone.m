function data = addone(data);

if (isa(data.Xtraining, 'gsingle') || isa(data.Xtraining, 'gdouble'))
  data.Xtraining = [data.Xtraining; gones(1, data.Ntraining)];
else
  data.Xtraining = [data.Xtraining; ones(1, data.Ntraining)];
end
  
if (isa(data.Xtest, 'gsingle') || isa(data.Xtest, 'gdouble'))
  data.Xtest = [data.Xtest; gones(1, data.Ntest)];
else
  data.Xtest = [data.Xtest; ones(1, data.Ntest)];
end
