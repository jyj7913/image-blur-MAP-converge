function psf = deblur_single_level(blurred, k_size)
    lambda_1 = 0.0006;
    alpha = 0.1;
    lambda_k = 0.001;
    num_iters = 20;

    psf = zeros(k_size)
    psf((k_size(1)+1)/2, (k_size(2)+1)/2) = 1;

    blurred = edgetaper(blurred, fspecial('gaussian', 60, 10));
    bx = imfilter(blurred, [0, -1, 1], 'corr', 'circular')
    by = imfilter(blurred, [0: -1: 1], 'corr', 'circular')
    blurred_g = cat(3, bx, by)

    for iter=1:num_iters
        latent_g = deconv_g(blurred_g, psf, lambda_1, alpha);
        [energy data prior_l prior_k] = energy_func(latent_g, blurred_g, psf, lambda_1, alpha, lambda_k);
        fprint('%d\t%f\t%f\t%f\t%f\n', iter, energy, data, prior_l, prior_k);
        psf = estimate_psf(blurred_g, latent_g, psf_size, lambda_k)
    end
end
