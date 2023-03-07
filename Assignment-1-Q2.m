%%
clear all;
clc;

%% Question # 2: Enhance the Quality of Image by Using Contrast Stretching
%  The equation of Contrast Stretching is "g = 1./(1 + (m./(f + eps)).^E)"
% The default value of "m=mean(im2double(Image))" and E=4.
% "E" controls the slope of function
% "g" is the Matlab implemetation of "s = T(r) = 1/1+(m/r)^E"

Img = imread('Lena.jpg');
Img = im2double(Img);


figure;
imshow(Img)

m = mean2(im2double(Img));
E = 5;                         % As mentioned above "4" is the default value
                               % The contrast-stretching performance is
                               % decreaseing below 4 and voice versa, but
                               % above '4' image becoming blur

Img2 = 1./(1 + (m./(Img + eps)).^E); % Equation of Contrast-Stretching 


imwrite(Img2, 'EnhancedImage.png');

figure;
imshow(Img2)

