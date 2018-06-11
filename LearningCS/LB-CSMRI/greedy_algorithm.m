%   This script contains runs the greedy algorithm by implementing the
%   given reconstruction algorithm in parallel until the desired rate.
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


p     = size(x_original,1);
multiple_training_rule = 'average';


if strcmp(orientation,'horizontal')
    rate_grid_1=ceil(ACS_lines_number/size(x_original,1)/0.025)*0.025;
elseif strcmp(orientation,'vertical')
    rate_grid_1=ceil(ACS_lines_number/size(x_original,1)/0.025)*0.025;
elseif strcmp(orientation,'2D_lines')
    ccc = zeros(size(x_original,1),size(x_original,2));
    ccc(p/2-ACS_lines_number/2+1:p/2+ACS_lines_number/2,p/2-ACS_lines_number/2+1:p/2+ACS_lines_number/2)=1;
    number_of_existing_elements = nnz(ccc);
    rate_grid_1=ceil(number_of_existing_elements/0.025/256/256)*0.025;
end

normalization = 'yes';
rate = rate+0.03;
rate_grid = rate_grid_1:0.025:rate;
ACS_type      = 'stable' ; 
ACS_shape     = 'full';    
warm_start    = 0;
ACS_size      = 0; 
aaa = zeros(size(x_original,1),size(x_original,2)); aaa(p/2-ACS_size/2+1:p/2+ACS_size/2,p/2-ACS_size/2+1:p/2+ACS_size/2)=1;ACS_indices = find(aaa) ;
rate_i = 0 ;
rate_i_vector = [];
selected_index_vector  = [];
mask_evolution = [];
max_objetive = 0;
objective_greedy = [];
mask = aaa;

if strcmp(orientation,'horizontal')
    
    label = zeros(1,p); 
    label(p/2-ACS_lines_number/2+1 : p/2 + ACS_lines_number/2 ) =1;
    mask(p/2-ACS_lines_number/2+1 : p/2 + ACS_lines_number/2,:)=1; % should generalize to all three case, do it later.

elseif  strcmp(orientation,'vertical')
    label = zeros(1,p); 
    label(p/2-ACS_lines_number/2+1 : p/2 + ACS_lines_number/2 ) =1;
    mask(:,p/2-ACS_lines_number/2+1 : p/2 + ACS_lines_number/2)=1; % should generalize to all three case, do it later.
    
elseif strcmp(orientation,'2D_lines')
    
    label = zeros(2,p); 
    label(1,p/2-ACS_lines_number/2+1 : p/2 + ACS_lines_number/2 ) =1;
    label(2,p/2-ACS_lines_number/2+1 : p/2 + ACS_lines_number/2 ) =1;
    mask(:,p/2-ACS_lines_number/2+1 : p/2 + ACS_lines_number/2)=1; % should generalize to all three case, do it later.
    mask(p/2-ACS_lines_number/2+1 : p/2 + ACS_lines_number/2,:)=1; % should generalize to all three case, do it later.
    
end


label_ACS               = zeros(1,p/2);  % when the ACS size is fixed, then it will not be changed
label_ACS(1:ACS_size/2) = 1;             % when the ACS size is fixed, then it will not be changed
previous_mask = mask;

%INITIAL PARAMETERS
x_0=zeros(size(x_original));

%%
kkk = 1;
counter_mask = 1;
while rate_i < rate
    greedy_iter = counter_mask
    
    label_elements     = find(label==0);         % rows/columns that are not yet added
    
    if strcmp(ACS_type,'dynamic')
        if strcmp(ACS_shape,'rings')
            label_elements_ACS = find(label_ACS==0);
        elseif strcmp(ACS_shape,'full')
            label_elements_ACS = find(label_ACS==0);
            label_elements_ACS = label_elements_ACS(1) ;
        end
        average_objective_value_vector           = zeros(numel(label_elements)+numel(label_elements_ACS),1);
        pixel_increases = zeros(numel(label_elements)+numel(label_elements_ACS),1);
    elseif strcmp(ACS_type,'stable')
        average_objective_value_vector           = zeros(numel(label_elements),1);
        pixel_increases =zeros(numel(label_elements),1);
        
    end
    
    if warm_start
        solution_array = zeros([size(x_original),numel(average_objective_value_vector)]); 
    end
    
    %%
    aba = tic; 
    parfor i = 1:numel(average_objective_value_vector) % parfor
        addpath(genpath('NESTA_v1.1/'))
        addpath(genpath('utilities/'))

        
        addpath(genpath('NESTA_v1.1/'))
        addpath(genpath('utilities/'))
        
        parfor_it = i
        
        if i > numel(label_elements)  % ACS part
            
            label_temp     = label;
            
            label_ACS_temp = label_ACS;
            label_ACS_temp( label_elements_ACS(i - numel(label_elements))  ) = 1;
            
            mask_temp =zeros(size(mask));
            
            if  strcmp(orientation,'horizontal')
                mask_temp(find(label_temp==1),:) = 1;
            elseif  strcmp(orientation,'vertical')
                mask_temp(:,find(label_temp==1)) = 1;
            elseif strcmp(orientation,'2D_lines')
                mask_temp(find(label_temp(1,:)==1),:) = 1;
                mask_temp(:,find(label_temp(2,:)==1)) = 1;
            end
            
            
            pixel_increases(i) = sum(sum(mask_temp - mask));
            
            
        else % vertical&/horizontal lines
            
            j               = label_elements(i);
            label_temp      = label;
            label_temp(j)   = 1;
            
            label_ACS_temp = label_ACS;
            
            mask_temp =zeros(size(mask));
            
            if  strcmp(orientation,'horizontal')
                mask_temp(find(label_temp==1),:) = 1;
            elseif  strcmp(orientation,'vertical')
                mask_temp(:,find(label_temp==1)) = 1;
            elseif strcmp(orientation,'2D_lines')
                mask_temp(find(label_temp(1,:)==1),:) = 1;
                mask_temp(:,find(label_temp(2,:)==1)) = 1;
            end
            
            
            pixel_increases(i) = sum(sum(mask_temp - mask));
            
        end
        
        %%
        ind = find(mask_temp(:)) ;
        
        average_objective_value = size(size(x_original,4),1); 
        x_BP_comp_all = zeros([size(x_original)]);
        for abc = 1:size(x_original,4)
              if warm_start
            other_param = [];
            other_param.x_0 = squeeze(x_0(:,:,:,abc));
              end
            abc;
            [x_BP_comp,x_original2]=reconstruction_algorithms(squeeze(x_original(:,:,:,abc)),ind,algo);
            x_BP_comp_all(:,:,1,abc) =     x_BP_comp;
            
            ERRORS = error_metric_FINAL(x_original2(ind_object{abc}),x_BP_comp(ind_object{abc}));
            average_objective_value(abc) = assign_objective_value(ERRORS,metric);
        end
        
        if strcmp(multiple_training_rule,'average')
            average_objective_value_vector(i) = mean(average_objective_value);
        end
        
        if warm_start
            solution_array(:,:,1,:,i) = x_BP_comp_all;
            x_BP_comp_all=[];
        end
        
        
    end
    one_parfor_time = toc(aba) ;
    
    %% 
    if strcmp(normalization,'yes')
        [~,selected_index]  =  max(  (average_objective_value_vector(:) - max_objetive )./pixel_increases(:)   );
    else
        [~,selected_index]  =  max(  (average_objective_value_vector(:) - max_objetive )   );
    end
    max_objetive  =   average_objective_value_vector(selected_index) ;
    
    if warm_start
        for abc = 1:size(x_original,4)
            x_0(:,:,:,abc) = solution_array(:,:,:,abc,selected_index);
        end
    end
    
    
    %% UPDATE OF MASK
    if selected_index > numel(label_elements)
        
        label_ACS( label_elements_ACS(selected_index - numel(label_elements))  ) = 1;
        mask = get_ACS_rings(p,label_ACS,ACS_shape);

        if  strcmp(orientation,'horizontal')
            mask(find(label==1),:) = 1;
        elseif  strcmp(orientation,'vertical')
            mask(:,find(label==1)) = 1;
        elseif strcmp(orientation,'2D_lines')
            mask(find(label(1,:)==1),:) = 1;
            mask(:,find(label(2,:)==1)) = 1;
        end
        
        
    else
        
        selected_index2=label_elements(selected_index);
        label(selected_index2) = 1;
        
         mask =zeros(size(x_original,1),size(x_original,2));
        
        if  strcmp(orientation,'horizontal')
            mask(find(label==1),:) = 1;
        elseif  strcmp(orientation,'vertical')
            mask(:,find(label==1)) = 1;
        elseif strcmp(orientation,'2D_lines')
            mask(find(label(1,:)==1),:) = 1;
            mask(:,find(label(2,:)==1)) = 1;
        end
        
    end
    
    rate_i = sum(mask(:))/numel(mask)
    mask_i = mask;
    
    
    %% It will save to exploit the nestedness property:
    
    if ( rate_i > rate_grid(kkk) ) && ( rate_i <= rate_grid(kkk+1) )
        
        rate_i_attained = rate_grid(kkk)
        kkk = kkk +1;
        mask = previous_mask;
        
        file_name_i = [file_name_base,'_algo=',algo,'_training=',training_image,(sprintf('_rate=%.3f',rate_i_attained)),'_orientation=',orientation,'_metric=',metric];
        
        save([file_name_i,'.mat'],'mask','objective_greedy','rate_i_vector','mask_evolution');
    end
    
    previous_mask = mask_i;
    objective_greedy = [objective_greedy;max_objetive];
    rate_i_vector = [rate_i_vector;rate_i];
    mask_evolution(:,:,counter_mask)  = mask_i;
    counter_mask = counter_mask +1
    
    % save(file_name_base,'mask_evolution','counter_mask','kkk','rate_i')
    
    
end
