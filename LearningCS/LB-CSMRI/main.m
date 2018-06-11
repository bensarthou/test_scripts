%   This script runs the LB-CSMRI greedy algoritm on a computing cluster in
%   parallel.
%
%   See our reference paper for the detailed explanation of the
%   algorithm.
%
%   B. Gözcü, R. K. Mahabadi, Y. H. Li, E. Ilıcak,  T. Çukur, J. Scarlett,
%   and V. Cevher. Learning-Based Compressive MRI. IEEE transactions
%   on medical imaging (2018)
%
%   Coded by: Baran Gözcü
%   École Polytechnique Fédérale de Lausanne, Switzerland.
%   Laboratory for Information and Inference Systems, LIONS.
%   contact: baran.goezcue@epfl.ch
%   Created: July 1, 2017
%   Last modified: April 20, 2018
%
% LB-CSMRIv1.0
% Copyright (C) 2018 Laboratory for Information and Inference Systems
% (LIONS), École Polytechnique Fédérale de Lausanne, Switzerland.
% This code is a part of LB-CSMRI toolbox.
% Please read COPYRIGHT before using this file.

%% training data
training_image = 'brain_training';
load(['data/',training_image]);

%% reconstruction algorithm
algo = 'TV'; % or you can add your recon algorithm into the function reconstruction_algorithm.m

%% paths
if exist('NESTA_v1.1')==0
    error('Please download NESTA solver from https://statweb.stanford.edu/~candes/nesta/ and place the NESTA_v1.1 folder in LB-CSMRI folder')
end
addpath(genpath('NESTA_v1.1/'))
addpath(genpath('utilities/'))


%% other parameters
file_name_base=['results/simulation_1'];
orientation   = 'horizontal'; % directions of phase encodes % vertical / horizontal / 2D_lines
rate = 0.30; % rate until which the algorithm is run
ACS_lines_number =0; % number of central lines that are selected beforehand
% objective function used in the greedy algorithm:
metric        = 'PSNR_abs'; % psnr computed on the absoluted valued images. (alternative: SSIM)

%% COMPUTING THE SUPPORT OF THE BRAINS ON WHICH THE
%  threshold paramater should be adjusted for different training data in order to
%  obtain the correct object support
threshold = 0.0012;

no_of_training = size(x_original,4);
p     = size(x_original,1);

ind_object = cell(no_of_training);
new_f_whole2 = zeros(size(x_original,1),size(x_original,2),no_of_training);
for ijk = 1:no_of_training
    x_original_i = (x_original(:,:,:,ijk));
    general_support_finder
    ind_object{ijk}=ind_object_i;
    new_f2 = zeros(size(x_original,1),size(x_original,2));new_f2(ind_object_i)=1;
    new_f_whole2(:,:,ijk) = new_f2;
end

sl_no=2; % to be shown
figure(32);subplot(122);imagesc(new_f_whole2(:,:,sl_no));title('support for error computation');colormap gray;axis off
subplot(121);imagesc(abs(x_original(:,:,1,sl_no)));title('original images');colormap gray;axis off


%% greedy algorithm will save the masks for different rates
greedy_algorithm
