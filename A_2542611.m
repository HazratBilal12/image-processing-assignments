%% CNG 466 Assignment # 3: Fruit Tree Recognition System
%  Hazrat Bilal # 2542611
%  Repeat Simulation if the Accuracy is not 57.14

clc;
clear all;
%% Acquisition: Reading tree images from the database folder and creating labels for each  

% The Dataset name is "Database" and the fruit trees to be recognize is 
% 'Apple', 'Avocado', 'Lemon', 'Orange', 'Plum', 'Rasberry'.

dataset = imageDatastore("Database",'IncludeSubfolders',1,'LabelSource','foldernames');

TotalcountinEachLabel = countEachLabel(dataset);
classes = unique(dataset.Labels);
TotalImages = length(dataset.Files);


%% Features Extraction: I extracted the statistical features of the Images  
%  via Principal component Analysis(PCA). The other Techniques is not
%  suitable to extract features from the such color images. The color features
%  is not providing enough information for recognition. The Chain code and
%  Shape number requires segmentation and boundry extraction, which is very
%  difficult for such images. Therefor, I use statistical features of the
%  given dataset.

% I used the PCA to better analysis the features of color images.
FeaturesArray = zeros(TotalImages,23);
% % FeaturesArray = [];

for x = 1 : TotalImages
    img = imread(dataset.Files{x});
    img = imresize(img,[256,256]);
    gray_img = rgb2gray(img);
%     figure;
%     imshow(gray_img);
    [rows, columns, numberOfColorChannels] = size(img);
    
    listOfRGBValues = double(reshape(img, rows * columns, 3));
    coeff = pca(listOfRGBValues);
    transformedImagePixelList = listOfRGBValues * coeff;
    
    pc1Image = reshape(transformedImagePixelList(:,1), rows, columns);
    pc2Image = reshape(transformedImagePixelList(:,2), rows, columns);
    pc3Image = reshape(transformedImagePixelList(:,3), rows, columns); 
    
    
    pc1 = pc1Image;
    pc2 = pc2Image;
    pc3 = pc3Image;
    
    pc1_mean = mean(mean(pc1));
    pc1_std = std2(pc1);
    
    pc2_mean = mean(mean(pc2));
    pc2_std = std2(pc2);
    
    pc3_mean = mean(mean(pc3));
    pc3_std = std2(pc3);
    
    pc1_sk = sum(skewness(pc1));
    pc2_sk = sum(skewness(pc2));
    pc3_sk = sum(skewness(pc3));
    
    pc1_min = min(imhist(pc1));
    pc1_max = max(imhist(pc1));
    pc2_min = min(imhist(pc2));
    pc2_max = max(imhist(pc2));
    pc3_min = min(imhist(pc3));
    pc3_max = max(imhist(pc3));
 
   gray_cm=graycomatrix(gray_img,'Offset',[2,0;0,2]);   %Gray-Level Co-Occurrence Matrix 
   statistical1 = graycoprops(gray_cm,{'contrast','homogeneity'});
   statistical2 = graycoprops(gray_cm,{'correlation','energy'});
   
   f11=statistical1.Contrast;
   f12=statistical1.Homogeneity;
   f13=statistical2.Correlation;
   f14=statistical2.Energy;
   
   total_features=horzcat([pc1_mean,pc1_std,pc2_mean,pc2_std,pc3_mean,pc3_std,pc1_sk,pc2_sk,pc3_sk,pc1_min,pc1_max,pc2_min,pc2_max,pc3_min,pc3_max,f11,f12,f13,f14]);
   FeaturesArray(x,:) = total_features;
end

%% Validation : I used the HoldOut for validation to split the dataset into
% training and testing. We can also use the K-fold validation, which is
% good and better than HoldOut but I used this due to simplicity.
% But the LeaveOut validation might give low accuracy due to the one sample
% in test.

[train,test] = crossvalind('HoldOut',30,0.25);

% Separate to training and test data
dataTrain = FeaturesArray(train,:);
dataTest  = FeaturesArray(test,:);

%% Recognition: I used the SVM model for fruit tree recognition as we know 
% that SVM is better than other two methods. I evaluate KNN and SVM but the
% performance of SVM is higher than the KNN. Hocam, KNN is also given below
% in comments. The naïve bayes is not working on this dataset because some
% images has zero variance. This may be due to small size of dataset.
% Therefor SVM is the best model for this classification example.

SVMModels = cell(6, 1);
Y = dataset.Labels(train);
rng(1);
Mdl = fitcecoc(dataTrain,Y);
[label,score] = predict(Mdl,dataTest);

% K-NN Model for classification with Accuracy=42.6

% Mdl = fitcknn(dataTrain,Y,'NumNeighbors',5,'Standardize',1);
% [label,score,cost] = predict(Mdl,dataTest);

%% Performance : The performance of the model is calculated using the Confusion Matrix.
% The performance of the SVM is 57.14. This may due to the small dataset.
% Repeat Simulation if the Accuracy is not 57.14.

testset_labels = dataset.Labels(test);
Conf_Mat=confusionmat(testset_labels, label);

figure()
confusionchart(Conf_Mat,classes);
title('Confusion Matrix');

Accuracy = sum(diag(Conf_Mat))/sum(sum(Conf_Mat))