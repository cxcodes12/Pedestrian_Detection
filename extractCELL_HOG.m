function [h] = extractCELL_HOG(amplit,theta)
%calcul histograma 9-bin pentru o celula de 8x8 pixeli
h = zeros(1,9);
for i=1:8
    for j=1:8
        a = amplit(i,j);
        t = theta(i,j);
        k = floor(abs(t/20-0.5)); %k de la 0 la 8
        ck = (k+0.5)*20;
        ck1 = (k+1.5)*20;
        if k==8
            h(9) = h(9)+a*(1-(t-ck)/20); 
            h(1) = h(1)+a*(t-ck)/20;
        else if k==0 && t<10
                h(1) = h(1)+a*(1+(t-ck)/20);
                h(9) = h(9)+abs(a*(t-ck)/20);
            else 
                h(k+1) = h(k+1)+a*(1-(t-ck)/20); 
                h(k+2) = h(k+2)+a*(t-ck)/20;
            end
        end
    end
end

end

