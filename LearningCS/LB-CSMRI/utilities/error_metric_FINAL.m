function output=error_metric_FINAL(f,I) % put varying number of arguments for the case 'clipped'



%% inputs
% f: original image
% I: reconstructed image

f=single(f); % it seems that it is already single
I=single(I);
N=numel(f);
    
    %% IMAGES:
    
    output.l2_error = norm(I-f,'fro')/norm(f(:),'fro');
    output.l2_error_abs = norm(abs(I)-abs(f),'fro')/norm(f(:),'fro');
    
    output.psnr      = 20*log10(max(abs(f(:)))*sqrt(N)/norm(I - f,'fro'));
    %output.psnr_abs   = psnr(abs(I),abs(f))  ;
    output.psnr_abs  = 20*log10(max(abs(f(:)))*sqrt(N)/norm(abs(I(:)) - abs(f(:)),'fro'));
    
    output.snr       = 20*log10((norm(f(:),'fro'))   /norm(I - f,'fro'));
    output.snr_abs   = 20*log10(norm(abs(f(:)),'fro')/norm(abs(I(:)) - abs(f(:)),'fro'));
 
    output.l1_error = norm((I)-(f),1)/norm((f(:)),1); 
    output.l1_error_abs = norm(abs(I)-abs(f),1)/norm(abs(f(:)),1); 
        
    %  SSIM
     output.ssim = ssim(  abs(I) ,  abs(f)  ,  'DynamicRange'  ,  max(abs(I(:))) - min(abs(I(:)))   ) ;

    
    
    
end