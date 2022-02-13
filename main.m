function LUT = make_LUT(lambda_l, alpha)
    % for every b find the optimal l
    % which minimizes |b-l|^2 + lambda_l |l|^alphav = -1:0.1/256:1;
    [b l] = meshgrid(v,v);
    tau = 0.01;
    w = max(abs(l),tau).^(alpha-2);
    prior_l = w.*l.^2;
    energies = (b-l).^2 + lambda_l * prior_l;
    [min_energies min_indices] = min(energies);LUT = [v; l(min_indices)];
end

function l = find_optimal_no_blur(blurred, lambda_l, alpha)
    LUT = make_LUT(lambda_l, alpha);
    bx = blurred(:,:,1);
    by = blurred(:,:,2);
    lx = interp1(LUT(1,:), LUT(2,:), bx(:), 'linear', 'extrap');
    lx = reshape(lx, size(bx));
    ly = interp1(LUT(1,:), LUT(2,:), by(:), 'linear', 'extrap');
    ly = reshape(ly, size(by));
    l = cat(3, lx, ly);
end

function psf = estimate_psf(blurred, latent, psf_size, lambda_k)
    B = fft2(blurred);
    L = fft2(latent);
    Bx = B(:,:,1);
    By = B(:,:,2);
    Lx = L(:,:,1);
    Ly = L(:,:,2);
    Lap = psf2otf([0 -1 0;-1 4 -1; 0 -1 0], [size(blurred,1), size(blurred,2)]);
    K = (conj(Lx).*Bx + conj(Ly).*By)./ (conj(Lx).*Lx + conj(Ly).*Ly + lambda_k.*Lap);
    psf = real(otf2psf(K, psf_size));
    psf(psf < max(psf(:))*0.05) = 0;
    psf = psf / sum(psf(:));
end

function x=conjgrad(x, b, n_iters, tol, Ax_func, func_param)
    r = b - Ax_func(x,func_param);
    p = r;
    rsold = sum(r(:).^2);
    for iter=1:n_iters
        Ap = Ax_func(p,func_param);
        alpha = rsold/sum(p(:).*Ap(:));
        x=x+alpha*p;
        r=r-alpha*Ap;
        rsnew=sum(r(:).^2);
        if sqrt(rsnew)<tol
            break;
        end
        p=r+rsnew/rsold*p;
        rsold=rsnew;
    end
end

function y = Ax(x, p)
    y = real(ifft2(conj(p.psf_f).* p.psf_f.*fft2(x)));
    y = y + p.lambda_l*p.weight.*x;
end

function latent = deconv_L2(blurred, latent, psf, lambda_l, weight, n_iters)
    img_size = size(blurred);
    psf_f = psf2otf(psf, img_size);
    b = real(ifft2(fft2(blurred).*conj(psf_f)));
    % run conjugate gradient
    p.lambda_l = lambda_l;
    p.psf_f = psf_f;
    p.weight = weight;
    latent = conjgrad(latent, b, n_iters, 1e-4, @Ax, p);
end

function latent = deconv_sps(blurred, psf, lambda_l, alpha, num_iters)
    tau = 0.01;
    if ~exist('num_iters', 'var')
        num_iters = 15;
    end
    % initial latent image
    B = fft2(blurred);
    K = psf2otf(psf, size(blurred));
    L = conj(K).*B./(conj(K).*K+lambda_l);
    latent = real(ifft2(L));
    % iterative update using IRLS
    for iter=1:num_iters
        w = max(abs(latent),tau).^(alpha-2);
        latent = deconv_L2(blurred,latent,psf,lambda_l,w,5);
    end
end

function [energy data prior_l, prior_k]=energy_func(latent, blurred, psf, lambda_1, alpha, lambda_k)
    tau=0.01;
    K = psf2otf(psf, size(blurred));
    b = real(ifft2(fft2(latent).*K));
    diff = b - blurred;
    data = sum(diff(:).^2);
    w = max(abs(latent),tau).^(alpha-2);
    prior_l = sum(w(:).*latent(:).^2);
    prior_k = sum(psf(:).^2);
    energy = data + lambda_l*prior_l + lambda_k*prior_k;
end

function psf = deblur_single_level(blurred, k_size)
    lambda_1 = 0.0006;
    alpha = 0.1;
    lambda_k = 0.001;
    num_iters = 20;

    psf = zeros(k_size);
    psf((k_size(1)+1)/2, (k_size(2)+1)/2) = 1;

    blurred = edgetaper(blurred, fspecial('gaussian', 60, 10));
    bx = imfilter(blurred, [0, -1, 1], 'corr', 'circular');
    by = imfilter(blurred, [0: -1: 1], 'corr', 'circular');
    blurred_g = cat(3, bx, by);

    for iter=1:num_iters
        latent_g = deconv_g(blurred_g, psf, lambda_1, alpha);
        [energy data prior_l prior_k] = energy_func(latent_g, blurred_g, psf, lambda_1, alpha, lambda_k);
        fprint('%d\t%f\t%f\t%f\t%f\n', iter, energy, data, prior_l, prior_k);
        psf = estimate_psf(blurred_g, latent_g, psf_size, lambda_k);
    end
end