function [HOG] = myHOGfeatures(im_gray)

im = double(im_gray);

%calcul gradienti
[M,N] = size(im);
dx = [-1 0 1];
dy = [-1; 0; 1];
Gx = zeros(M,N);
Gy = zeros(M,N);
for i=1:M
    for j=2:N-1
        Gx(i,j) = sum(sum(dx.*im(i,j-1:j+1)));
    end
end
for i=2:M-1
    for j=1:N
        Gy(i,j) = sum(sum(dy.*im(i-1:i+1,j)));
    end
end
%amplit si orientarea gradientilor
G = sqrt(Gx.^2+Gy.^2);
theta = abs(rad2deg(atan2(Gy,Gx)));
%corectie unghiuri negative
theta(theta<0) = theta(theta<0)+180;

%impartire img in celule 8x8 si calcul histograme concatenate pt 4x4 celule
HOG = [];
for i=1:8:M-8
    for j=1:8:N-8
        G_4cell = G(i:i+15,j:j+15);
        theta_4cell = theta(i:i+15,j:j+15);
        h_block = [];
        for ii=1:8:9
            for jj=1:8:9
                G_current = G_4cell(ii:ii+7,jj:jj+7);
                theta_current = theta_4cell(ii:ii+7,jj:jj+7);
                h_cell = extractCELL_HOG(G_current,theta_current);
                h_block = [h_block, h_cell];
            end
        end
        h_block = h_block./norm(h_block,2);
        HOG = [HOG,h_block];
    end
end

end