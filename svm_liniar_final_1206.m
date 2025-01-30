clc
clearvars
close all

%setul de date pozitive (pietoni)
pos_path = 'path for INRIAPerson\train_64x128_H96\pos';
imds_pos = imageDatastore(pos_path);
%setul de date negative (lipsa pietoni)
neg_path = 'path for INRIAPerson\train_64x128_H96\neg_64x128';
imds_neg = imageDatastore(neg_path);

%extragere trasaturi hog pt pietoni
hog_vector_initial = [];
for i=1:size(imds_pos.Files,1)
    im = readimage(imds_pos,i);
    im = rgb2gray(im);
    hog_feat = extractHOGFeatures(im,'CellSize',[8,8],'BlockSize',[2,2]);
    hog_vector_initial = [hog_vector_initial; hog_feat];
end
etichete_initial = ones(size(imds_pos.Files,1),1);

%extragere trasaturi hog pt fundal
for i=1:size(imds_neg.Files,1)
    im = readimage(imds_neg,i);
    im = rgb2gray(im);
    hog_feat = extractHOGFeatures(im,'CellSize',[8,8],'BlockSize',[2,2]);
    hog_vector_initial = [hog_vector_initial; hog_feat];
end
etichete_initial = [etichete_initial; zeros(size(imds_neg.Files,1),1)];

save('hog_vector_initial','hog_vector_initial');
save('etichete_initial','etichete_initial');


%% antrenare SVM linear
clc
modelv1 = fitcsvm(hog_vector_initial,etichete_initial,'OptimizeHyperparameters','auto', ...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName', ...
    'expected-improvement-plus'))
save('modelv1','modelv1');
modelv1_prob = fitPosterior(modelv1, hog_vector_initial,etichete_initial);
save('modelv1_prob','modelv1_prob');


%% hard-mining
clc
fullneg_path = 'path for INRIAPerson\train_64x128_H96\neg';
imds_hm = imageDatastore(fullneg_path); %hard-mining
hog_hm = [];
prob_hm = [];
nrSW = 0;
for nr=1:size(imds_hm.Files,1)
    disp(['Procent parcurgere cele 1218img:  ',num2str(nr/size(imds_hm.Files,1)*100),'%']);
    im = readimage(imds_hm,nr);
    %creare piramida variabila
    [M,N,~] = size(im);
    k1 = ceil(128/M*10)/10;
    k2 = ceil(4*128/M*10)/10;
    scale = linspace(k1,k2,7); %7 scari
    for k=1:length(scale)
        im_rsz = imresize(im,scale(k));
        im_rsz = rgb2gray(im_rsz);
        [m,n] = size(im_rsz);
        for i=1:round(15*scale(k)):m-127
            for j=1:round(10*scale(k)):n-63
                nrSW = nrSW+1;
                sw = im_rsz(i:i+127,j:j+63);
                hog_sw = extractHOGFeatures(sw,'CellSize',[8,8],'BlockSize',[2,2]);
                [pred, prob] = predict(modelv1_prob,hog_sw);
                if pred==1
                    hog_hm = [hog_hm; hog_sw];
                    prob_hm = [prob_hm; prob(2)];
                end
            end
        end
    end
    disp(['Sliding windows FP detectate pana acum: ', num2str(size(hog_hm,1))]);
    disp(['Procent FP pana acum: ', num2str(size(hog_hm,1)/nrSW*100),'%']);
    disp(' ');
end
save('hog_hm','hog_hm');
save('prob_hm','prob_hm');
%% reantrenare model
hog_hm_train = hog_hm(prob_hm>=0.7,:);
hog_addhm = [hog_vector_initial; hog_hm_train];
etichete_addhm = [etichete_initial; zeros(size(hog_hm_train,1),1)];

modelv2 = fitcsvm(hog_addhm,etichete_addhm,'OptimizeHyperparameters','auto', ...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName', ...
    'expected-improvement-plus'))
save('modelv2','modelv2');

modelv2_prob = fitPosterior(modelv2, hog_addhm,etichete_addhm);
save('modelv2_prob','modelv2_prob');

%% evaluare performante clasificare imagini pozitive/negative pe img 128x64
clc
imds_neg = imageDatastore('path for INRIAPerson\test_64x128_H96\neg_128x64');
imds_poz = imageDatastore('path for INRIAPerson\test_64x128_H96\poz_128x64');
predictii = [];
for i=1:length(imds_poz.Files)
    im = readimage(imds_poz,i);
    im = imresize(im,[128,64]);
    im = rgb2gray(im);
    hog_p = extractHOGFeatures(im,'CellSize',[8,8],'BlockSize',[2,2]);
    pred = predict(modelv2_prob,hog_p);
    predictii = [predictii, pred];
end

for i=1:length(imds_neg.Files)
    im = readimage(imds_neg,i);
    im = rgb2gray(im);
    hog_n = extractHOGFeatures(im,'CellSize',[8,8],'BlockSize',[2,2]);
    pred = predict(modelv2_prob,hog_n);
    predictii = [predictii, pred];
end

ground_truth = [ones(1,length(imds_poz.Files)), zeros(1,length(imds_neg.Files))];

accuracy = sum(predictii==ground_truth)/length(ground_truth)*100;
    

%% testare pe imagine noua
clc
close all


%deschid fisierul cu caile catre adnotari (.txt)
fid = fopen('path for INRIAPerson\Test\annotations.lst', 'r');
data = textread('path for INRIAPerson\Test\annotations.lst', '%s', 'delimiter', '\n');
fclose(fid);
%initializare parametri

TP_04=0; FP_04=0; FN_04=0;
TP_05=0; FP_05=0; FN_05=0;

for idx_img_test = 1:length(data)
img_name = [];
[~,img_name] = fileparts(data{idx_img_test});
img_name = [img_name,'.png'];
full_path_img = fullfile('path for INRIAPerson\Test\pos',img_name);
bboxes = [];
scores = [];


im_orig = imread(full_path_img);
RESIZE_FACTOR = 0.5;
im = imresize(im_orig, RESIZE_FACTOR);
%creare piramida variabila
[M,N,~] = size(im);
k1 = ceil(128/M*10)/10;
k2 = ceil(4*128/M*10)/10;
scale = linspace(k1,k2,7); %7 scari
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
% 
% figure, imshow(im);
% for i=1:size(bb,1)
%     rectangle('Position',[bb(i,1),bb(i,2),bb(i,3),bb(i,4)],'EdgeColor','g'); hold on
% end

%parcurg fisierul cu adnotari si citesc fiecare linie (fiecare linie reprezinta fisierul cu adnotari pt imaginea corespondenta)

file_name = fullfile('path for INRIAPerson',data{idx_img_test});
fileID = fopen(file_name, 'r'); %deschidere .txt in mod citire
%citire linie cu linie si cautare format bounding box
bb_gt = [];
tline = fgetl(fileID);
while ischar(tline) %pana cand linia citita e goala
    if contains(tline, 'Bounding box') 
        bbox = regexp(tline, '\((\d+), (\d+)\) - \((\d+), (\d+)\)', 'tokens');
        xmin = str2double(bbox{1}{1});
        ymin = str2double(bbox{1}{2});
        xmax = str2double(bbox{1}{3});
        ymax = str2double(bbox{1}{4});  
        bb_gt = [bb_gt; xmin,ymin,xmax,ymax];
    end
    tline = fgetl(fileID);
end
fclose(fileID);


%citire bounding boxes corespondente
bb_pred = bb; %format [xmin,ymin,width,height]
isChecked_pred_04 = zeros(size(bb_pred,1),1);
isChecked_gt_04 = zeros(size(bb_gt,1),1);

isChecked_pred_05 = zeros(size(bb_pred,1),1);
isChecked_gt_05 = zeros(size(bb_gt,1),1);

if size(bb_pred,1)~=0 || size(bb_gt,1)~=0
    if size(bb_pred,1)==0 && size(bb_gt,1)~=0
        FN_05 = FN_05+size(bb_gt,1);
        FN_04 = FN_04+size(bb_gt,1);
    else if size(bb_pred,1)~=0 && size(bb_gt,1)==0
            FP_05 = FP_05+size(bb_pred,1);
            FP_04 = FP_04+size(bb_pred,1);
        else 
            for i=1:size(bb_pred,1)

                box_curent = bb_pred(i,:);
                detFlag_04 = 0; %il fac 1 daca gasesc o suprapunere(detectie corecta)
                detFlag_05 = 0;
                for j=1:size(bb_gt,1)

                    gt_curent = bb_gt(j,:);
                    xmin_gt = gt_curent(1);
                    ymin_gt = gt_curent(2);
                    xmax_gt = gt_curent(3);
                    ymax_gt = gt_curent(4);
                    %obtin coord colturi opuse ale predictiei
                    xmin_pred = box_curent(1);
                    xmax_pred = box_curent(1)+box_curent(3);
                    ymin_pred = box_curent(2);
                    ymax_pred = box_curent(2)+box_curent(4);
                    %calculez IOU
                    inter_area = max(0, min(xmax_pred, xmax_gt) - max(xmin_pred, xmin_gt)) * max(0, min(ymax_pred, ymax_gt) - max(ymin_pred, ymin_gt));
                    union_area = (xmax_pred - xmin_pred) * (ymax_pred - ymin_pred) + (xmax_gt - xmin_gt) * (ymax_gt - ymin_gt) - inter_area;
                    iou = inter_area / union_area;
                    if iou>=0.4
                        detFlag_04 = 1;
                        isChecked_gt_04(j) = 1;
                        isChecked_pred_04(i) = 1;
                    end

                    if iou>=0.5
                        detFlag_05 = 1;
                        isChecked_gt_05(j) = 1;
                        isChecked_pred_05(i) = 1;
                    end
                end
                if detFlag_05==1
                    TP_05 = TP_05+1;
                end
                if detFlag_04==1
                    TP_04 = TP_04+1;
                end

            end
            FP_04 = FP_04+sum(isChecked_pred_04==0);
            FN_04 = FN_04+sum(isChecked_gt_04==0);

            FP_05 = FP_05+sum(isChecked_pred_05==0);
            FN_05 = FN_05+sum(isChecked_gt_05==0);
        end
    end
end
           
figure, imshow(im_orig);
for i=1:size(bb,1)
    rectangle('Position',[bb(i,1),bb(i,2),bb(i,3),bb(i,4)],'EdgeColor','g','LineWidth',1.5); hold on
end
for i=1:size(bb_gt,1)
    xmin = bb_gt(i,1);
    ymin = bb_gt(i,2);
    width = bb_gt(i,3)-bb_gt(i,1);
    height = bb_gt(i,4)-bb_gt(i,2);
    rectangle('Position',[xmin,ymin,width,height],'EdgeColor','r','LineWidth',1.5); hold on
end
% TP
% FN
% FP
precision_05 = TP_05/(TP_05+FP_05+eps)*100;
recall_05 = TP_05/(TP_05+FN_05+eps)*100;
F1_score_05 = 2*precision_05*recall_05/(precision_05+recall_05+eps);
disp(['Procent imagini parcurse: ',num2str(idx_img_test/length(data)*100),'%']);
disp(['Precizie actuala: ', num2str(precision_05),'%']);
disp(['Recall actual: ',num2str(recall_05),'%']);
disp(['F1-score actual: ',num2str(F1_score_05),'%']);
disp(' ');

end

precision_04 = TP_04/(TP_04+FP_04+eps)*100;
recall_04 = TP_04/(TP_04+FN_04+eps)*100;
F1_score_04 = 2*precision_04*recall_04/(precision_04+recall_04+eps);



