function [sbatch2 sbatch3] = select_neighbors(sbatch1, data)

l = double(data.Ltraining(sbatch1)) + 1;
nnn1 = size(data.indnlabels,1);
sbatch2 = double(data.indnlabels(ceil(rand(numel(sbatch1),1).*(data.nnlabels(l))) + (l-1)*nnn1));

nnn2 = size(data.nnTraining,1);
sbatch3 = double(data.nnTraining(ceil(rand(numel(sbatch1),1).*(data.nns(sbatch1))) + (sbatch1-1)*nnn2));
