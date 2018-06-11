function y=assign_objective_value(ERRORS,metric)

if strcmp(metric,'PSNR_abs')
    y     =    ERRORS.psnr_abs;
end

if strcmp(metric,'PSNR')
    y     =    ERRORS.psnr;
end


if strcmp(metric,'SSIM')
    y     =   ERRORS.ssim;
end


if strcmp(metric,'L1')
    y     = - ERRORS.l1_error;
end

if strcmp(metric,'L2')
    y     =  - ERRORS.l2_error;
end

if strcmp(metric,'L1_abs')
    y     = - ERRORS.l1_error_abs;
end

if strcmp(metric,'L2_abs')
    y     =  - ERRORS.l2_error_abs;
end


end


