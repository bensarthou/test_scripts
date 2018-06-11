%% support finder
 
 im_sqr = abs(x_original_i)/norm(x_original_i,'fro');

 bw = im2bw(im_sqr,threshold);
 ground_truth = bw;

%% Convex Hull
 ground_truth2 = bwconvhull((ground_truth)) ;
 f=ground_truth2;

 % YOU CAN COMPUTE THE ERROR ON THESE INDICES ONLY:
 ind_object_i = find(f);

%%
% figure
% subplot(121)
% imagesc(abs(x_original_i));title('original')
% subplot(122)
% imagesc(f);title('indices on which the error should be computed')
% 
% 
% 
%     


    