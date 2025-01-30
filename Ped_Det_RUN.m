clc
clearvars
close all

im_orig = imread('person_200.png');
RESIZE_FACTOR = 0.5;
im = imresize(im_orig, RESIZE_FACTOR);
load('modelv2_prob.mat');

%creare piramida variabila
[M,N,~] = size(im);
k1 = ceil(128/M*10)/10;
k2 = ceil(4*128/M*10)/10;
scale = linspace(k1,k2,7); %7 scale

bboxes = [];
scores = [];
for k=1:length(scale)
     im_rsz = imresize(im,scale(k));
     im_rsz = rgb2gray(im_rsz);
     [m,n] = size(im_rsz);
     for i=1:round(scale(k)*10):m-127
         for j=1:round(scale(k)*5):n-63
             sw = im_rsz(i:i+127,j:j+63);
             hog_sw = extractHOGFeatures(sw,'CellSize',[8,8],'BlockSize',[2,2]);
             [class,score] = predict(modelv2_prob,hog_sw);
             if class==1 
                 x = round(j/scale(k));
                 y = round(i/scale(k));
                 w = floor(64/scale(k));
                 h = floor(128/scale(k));
                 bboxes = [bboxes; x,y,w,h];
                 scores = [scores; score(2)];
             end
         end
     end
end
% figure, imshow(im);
% for i=1:size(bboxes,1)
%     rectangle('Position',[bboxes(i,1),bboxes(i,2),bboxes(i,3),bboxes(i,4)],'EdgeColor','g'); hold on
% end

% non maximum suppression
bboxes = round(bboxes/RESIZE_FACTOR);
relevantBoxes = nms(bboxes,0.5,scores);

% figure, imshow(im);
% for i=1:size(relevantBoxes,1)
%     rectangle('Position',[relevantBoxes(i,1),relevantBoxes(i,2),relevantBoxes(i,3),relevantBoxes(i,4)],'EdgeColor','g'); hold on
% end


% eliminare bboxes redundante
bb = relevantBoxes;
nr_bb = size(bb,1);
i=1;
while (i<=nr_bb)
    box = bb(i,:);
    j=1;
    %verific care e la interior
    while(j<=nr_bb)
        boxsec = bb(j,:);
        if(box(1)<boxsec(1) & box(2)<boxsec(2) & box(3)>=boxsec(3) & box(4)>=boxsec(4) & box(1)+box(3)>=boxsec(1)+boxsec(3) & box(2)+box(4)>=boxsec(2)+boxsec(4))
            box_centru = [box(1)+box(3)/2, box(2)+box(4)/2];
            boxsec_centru = [boxsec(1)+boxsec(3)/2, boxsec(2)+boxsec(4)/2];
            if abs(box_centru(1) - boxsec_centru(1))<=0.25*box(3) || abs(box(2)-boxsec(2))<=0.25*box(4) %verificare distanta dintre centre bboxes suprapuse pe axa x sau pe y
                bb(j,:) = [];
                nr_bb = size(bb,1);
            end
        end
        j = j+1;
    end
    i = i+1;
end
  
%micsorare bb
for i=1:size(bb,1)
    bb(i,1) = round(bb(i,1)+0.125*bb(i,3));
    bb(i,2) = round(bb(i,2)+0.1*bb(i,4));
    bb(i,3) = floor(0.75*bb(i,3));
    bb(i,4) = floor(0.8*bb(i,4));
end

figure, imshow(im_orig);
for i=1:size(bb,1)
    rectangle('Position',[bb(i,1),bb(i,2),bb(i,3),bb(i,4)],'EdgeColor','g','LineWidth',1.5); hold on
end

