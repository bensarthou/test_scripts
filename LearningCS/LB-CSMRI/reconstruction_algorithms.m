function [x_recon,x_original]=reconstruction_algorithms(x_original,ind,algo)

%   This script contains the reconstruction algorithm for which the k-space
%   mask will be optimized. You can add your reconstruction algorithm below 
%   in order to obtain a mask for an improved performance of your
%   reconstruction algorithm
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



if strcmp(algo,'TV')
    
    opts        = [];
    opts.maxiter = 20000;
    muf = 1e-5;
    opts.Verbose  = 0;
    opts.MaxIntIter = 1;
    opts.TolVar = 1e-5;
    opts.typemin = 'tv';
    delta= 0;

    
    [nx ny]=size(x_original);
    N=nx*ny;
    
    measurement_forward  = @(x) fft2fwd(reshape(x,[nx,ny]),ind);
    measurement_backward = @(x) reshape(fft2adj(x,ind,nx,ny),[N,1]);
    
    b  = measurement_forward(x_original);
    A=measurement_forward;
    At=measurement_backward;
    
    [x_NESTA_complex] =NESTA(A,At,b,muf,delta,opts);
    x_recon=reshape(x_NESTA_complex,[nx,ny]);
    
elseif strcmp(algo,'my_favorite_recon_algorithm_1')
    % you can add other reconstruction algorithms here
elseif strcmp(algo,'my_favorite_recon_algorithm_2')
    % you can add other reconstruction algorithms here
    
end





end
