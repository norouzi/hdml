%% For CUDA functionality before running matlab, run
% export LD_LIBRARY_PATH=/pkgs_local/cuda-4.0/lib64/
% export PATH=$PATH:/pkgs_local/cuda-4.0/bin/
% export CUDA_PATH='/pkgs_local/cuda-4.0';

%% If running multiple instances of code on the same machine the number of threads for openmp should be restricted by
% export OMP_NUM_THREADS=4;

%% For Jacket GPU functionality
% addpath ~norouzi/research/jacket/engine;
% ginfo      % shows the gpu information
% gselect(1) % selects the first gpu

% Compilation of the mex files
mex loss_aug_inf_triplet_mex0.cpp CXXFLAGS="\$CXXFLAGS -Wall -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"
mex loss_aug_inf_mex0.cpp CXXFLAGS="\$CXXFLAGS -Wall -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"
mex utils/hammknn_mex.cpp utils/linscan.cpp CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp";

% adding some stuff to the path.
addpath utils; % You should be at thre root of the project to run this line.
addpath /YAEL_HOME/matlab/ % download and install yael library and change this line


% parameters & RUN2
DATA = 'mnist';  % either 'mnist' or 'cifar10'
REG = 1;         % always 1 -- the term that enforces bits to have a mean of zero
NB = 32;         % number of bits
LOSS = 1;        % loss type: 0 -> pairwise hinge / 1 -> triplet ranking
LSCALE = 1;      % An scaling parameter multiplied by the loss
NN = 1000;       % Number of same-class items to be used in the positive set for training.
                 % NN=Inf -> all of the items from the same class
                 % NN=1000 -> find same-class 1000-NN in euclidean distance and set them as target positive examples for each item
GPU = 0;         % GPU=0 -> No GPU / GPU=1 -> GPU (change this to 1 if you have jacket configured)
SHRINK_W = [3e-2 3e-3 3e-4];
                 % weight decay parameter -- cross validating on this parameter on 3 candidates
RUN2             % given the above parameters runs the code


% Linear hash functions on CIFAR-10
DATA = 'cifar10'; REG = 1; NB = 128; LOSS = 1; LSCALE = 1; NN = Inf; GPU = 0; SHRINK_W = 1e-3;
RUN2


% Neural Net because NB is two dimensional. Calles compute_NN_output and compute_NN_grad
DATA = 'mnist'; REG = 1; NB = [512 32]; LOSS = 1; LSCALE = 1; NN = Inf; GPU = 0; SHRINK_W = 1e-4*[1; 1];
RUN2
