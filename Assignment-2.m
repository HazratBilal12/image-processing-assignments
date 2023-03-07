%% Assignment 2: Image Restoration and Reconstruction
clear all; 
clc;
%% Step1: Identification of noise via experiment or visual inspection

% Image 1: "noisy1.png"

Img1 = imread("noisy1.png");
imshow(Img1);



% Hocam Honestly, I did not visually identify the noise, but It seems to be
% an image corrupted by horizental scanning lines due to the illusion caused 
% by the strips on the cat body(Periodic Noise).

% This is my assumption, now I will investigate the spatial domain and then
% Fourier Spectrum of this image to check wether I am wright or wrong.


% I will take a solid area which is uniformly illuminated. Then, the noise 
% can be estimated by analyzing the histogram of this area. 
 
% I cropped a rectangular part of the image for understanding the histogram
% and image noise by using a Mathworks "drawrectangle()" command. I can
% also do it by the method discussed in Lecture-6 Exercise. I chose the
% Mathswork commad.


figure;
imshow(Img1);
title("Please Select a Solid Region with Uniform Illumination :)");

% Following "drawretangle()" command is taken from Mathworks. 
% https://www.mathworks.com/help/images/ref/drawrectangle.html#mw_275b6703-1df2-4d16-aed6-cdfcf02ad8af
% I check different regions of the image from top and bottom corners. The
% noise(Uniform) is same in every solid area from different region.


% "Position" of the ROI, specified as a 1-by-4 numeric vector of the form [xmin, ymin, width, height].
% xmin and ymin specify the coordinates of the upper left corner of the rectangle. 
% width and height specify the width and height of the rectangle and must be nonnegative.


Img1_rect = drawrectangle();
Location = Img1_rect.Position;

x1 = Location(2);
y1 = Location(1);

x2 = round(x1 + Location(4));
y2 = round(y1 + Location(3));

Img1_cropped = Img1(x1:x2 , y1:y2);



figure;
subplot(1,2,1); imshow(Img1_cropped);
title("noisy1.png Strip for Noise Identification");

subplot(1,2,2); histogram(Img1_cropped);
title("noisy1.png Strip Histogram");


figure;
subplot(2,3,1);imshow(Img1);
title(" Given Noisy Image(noisy1.png)");

subplot(2,3,2);histogram(Img1);
title("Noisy1 Histogram");


% Size of Given Image (noisy1.png):
M = size(Img1,1);      
N = size(Img1,2);      

% Fourier Transform of Given Image (noisy1.png):

FT = fft2(Img1);

% Centered Fourier Transform
Center_FT = fftshift(FT);

% Use the abs function to compute the magnitude of the combined components

% In Fourier transforms, high peaks are so high they hide details. 
% Reduce contrast with the log function

% %Fourier spectrum we want to look at

subplot(2,3,3);imshow(abs(log(1 + Center_FT)),[]);
title("Fourier Spectrum");


% It is clear from the histogram of Image that the noise is uniform noise.
% The historam is similar to the PDF of Uniform noise and Fourier 
% Spectrum does not help me very well to identify the noise. Which points  
% that the noise is additive not periodic.  

% Step2: Removal of the Identified Noise

% It is clear from the image and it's histogram that seems to be
% similar to Uniform noise [(The historgram lies in range of pepper noise {(Because
% background is completely black)} but visually there is no black dots on
% the cats)] which is spatially independent 


% Midpoint Filter

% Hocam I took this method of implementation of Midpoint filter from Book
% page 231.

% I removed the noise using Midpoint fitler according to slide # 33 It work
% best for Uniform Noise.

% Hocam I am using direct functions for Filters and Edge detection(Sobel:slide#65 Lec#3)
% because of your permission to use it. I can implement it using indirect method.

f1 = ordfilt2(Img1, 1, ones(3,3), 'symmetric');
f2 = ordfilt2(Img1, 9, ones(3,3), 'symmetric');
f = imlincomb(0.5, f1, 0.5, f2);

figure;
imshow(f);
title("Please Select a Solid Region with Uniform Illumination :)");

f_rect = drawrectangle();
pos = f_rect.Position;

x1 = pos(2);
y1 = pos(1);

x2 = round(x1 + pos(4));
y2 = round(y1 + pos(3));

f_cropped = f(x1:x2 , y1:y2);

subplot(1,3,1);imshow(f_cropped);
title("Image1 Strip after denoising");

subplot(1,3,2);histogram(f_cropped);
title("Strip Histogram after denoising");

subplot(1,3,3);histogram(f);
title("Histogram after denoising (Midpoint filter)");

figure;
imshow(f);
title("Midpoint Filtered Image");

imwrite(f,"recovered1.png")


% It is clear from the Edges that the Edges became clear after denoising
% the image. 

Img1e =edge(Img1,'sobel');
figure;
imshow(Img1e);
title("Combined-edge Before Denoising noisy1.png");

fe =edge(f,'sobel');
figure;
imshow(fe);
title("Combined-edge After Denoising using Midpoint Filter");




% The Idirect method for using Edge Detection

Img1_IE = int32(Img1);

Zeros_x = int32(zeros(M+2,N+2));
Zeros_y = int32(zeros(M+2,N+2));

Hx = [-1,-2,-1,0,0,0,1,2,1]; %derivative in x-direction  % Sobel filter
Hy = [-1,0,1,-2,0,2,-1,0,1]; %derivative in y-direction

%convolution for x-direction edge

for i=2:size(Img1_IE,1)-1
    for j=2:size(Img1_IE,2)-1
        %temp = 0;
        temp = int32(0);
        temp = temp + Hx(1)*Img1_IE(i-1,j-1);
        temp = temp + Hx(2)*Img1_IE(i-1,j);
        temp = temp + Hx(3)*Img1_IE(i-1,j+1);
        temp = temp + Hx(4)*Img1_IE(i,j-1);
        temp = temp + Hx(5)*Img1_IE(i,j);
        temp = temp + Hx(6)*Img1_IE(i,j+1);
        temp = temp + Hx(7)*Img1_IE(i+1,j-1);
        temp = temp + Hx(8)*Img1_IE(i+1,j);
        temp = temp + Hx(9)*Img1_IE(i+1,j+1);

       
        Zeros_x(i,j) = temp;
        
    end
end

%convolution for y-direction edge

for i=2:size(Img1_IE,1)-1
    for j=2:size(Img1_IE,2)-1
        %temp = 0;
        temp = int32(0);
        temp = temp + Hy(1)*Img1_IE(i-1,j-1);
        temp = temp + Hy(2)*Img1_IE(i-1,j);
        temp = temp + Hy(3)*Img1_IE(i-1,j+1);
        temp = temp + Hy(4)*Img1_IE(i,j-1);
        temp = temp + Hy(5)*Img1_IE(i,j);
        temp = temp + Hy(6)*Img1_IE(i,j+1);
        temp = temp + Hy(7)*Img1_IE(i+1,j-1);
        temp = temp + Hy(8)*Img1_IE(i+1,j);
        temp = temp + Hy(9)*Img1_IE(i+1,j+1);

        
        Zeros_y(i,j) = temp;
        
    end
end



Image1_IE = uint8(abs(Zeros_x) + abs(Zeros_y));

figure;
imshow(Image1_IE);
title("Combined-edge Before Removal of Noise using Indirect Method");





fead = int32(f);

Zeros_x = int32(zeros(M+2,N+2));
Zeros_y = int32(zeros(M+2,N+2));

Hx = [-1,-2,-1,0,0,0,1,2,1]; %derivative in x-direction  
Hy = [-1,0,1,-2,0,2,-1,0,1]; %derivative in y-direction

%convolution for x-direction edge

for i=2:size(fead,1)-1
    for j=2:size(fead,2)-1
        %temp = 0;
        temp = int32(0);
        temp = temp + Hx(1)*fead(i-1,j-1);
        temp = temp + Hx(2)*fead(i-1,j);
        temp = temp + Hx(3)*fead(i-1,j+1);
        temp = temp + Hx(4)*fead(i,j-1);
        temp = temp + Hx(5)*fead(i,j);
        temp = temp + Hx(6)*fead(i,j+1);
        temp = temp + Hx(7)*fead(i+1,j-1);
        temp = temp + Hx(8)*fead(i+1,j);
        temp = temp + Hx(9)*fead(i+1,j+1);

       
        Zeros_x(i,j) = temp;
        
    end
end

%convolution for y-direction edge

for i=2:size(fead,1)-1
    for j=2:size(fead,2)-1
        %temp = 0;
        temp = int32(0);
        temp = temp + Hy(1)*fead(i-1,j-1);
        temp = temp + Hy(2)*fead(i-1,j);
        temp = temp + Hy(3)*fead(i-1,j+1);
        temp = temp + Hy(4)*fead(i,j-1);
        temp = temp + Hy(5)*fead(i,j);
        temp = temp + Hy(6)*fead(i,j+1);
        temp = temp + Hy(7)*fead(i+1,j-1);
        temp = temp + Hy(8)*fead(i+1,j);
        temp = temp + Hy(9)*fead(i+1,j+1);

        
        Zeros_y(i,j) = temp;
        
    end
end




f_IE = uint8(abs(Zeros_x) + abs(Zeros_y));

figure;
imshow(f_IE);
title("Combined-edge After Removal of Noise Using Indirect Method");



%% Step1: Identification of noise via experiment or visual inspection

% Image 2: "noisy2.png"

Img2 = imread("noisy2.png");
imshow(Img2);


% Hocam, Visually, The image seems to be corrupted by by Salt noise. 

% This is my assumption, now I will investigate the spatial domain and then
% Fourier Spectrum of this image to check wether I am wright or wrong.


% I will take a solid area which is uniformly illuminated. Then, the noise 
% can be estimated by analyzing the histogram of this area. 
 
% I cropped a rectangular part of the image for understanding the histogram
% and image noise by using a Mathworks "drawrectangle()" command. I can
% also do it by the method discussed in Lecture-6 Exercise. I chose the
% Mathswork commad.


figure;
imshow(Img2);
title("Please Select a Solid Region with Uniform Illumination :)");


% Following "drawretangle()" command is taken from Mathworks. 
% https://www.mathworks.com/help/images/ref/drawrectangle.html#mw_275b6703-1df2-4d16-aed6-cdfcf02ad8af
% I check different regions of the image from top and bottom corners. The
% noise(Uniform) is same in every solid area from different region.


% "Position" of the ROI, specified as a 1-by-4 numeric vector of the form [xmin, ymin, width, height].
% xmin and ymin specify the coordinates of the upper left corner of the rectangle. 
% width and height specify the width and height of the rectangle and must be nonnegative.


Img2_rect = drawrectangle();
Location = Img2_rect.Position;

x1 = Location(2);
y1 = Location(1);

x2 = round(x1 + Location(4));
y2 = round(y1 + Location(3));

Img2_cropped = Img2(x1:x2 , y1:y2);


figure;
subplot(1,2,1); imshow(Img2_cropped);
title("noisy2.png Strip for Noise Identification");

subplot(1,2,2); histogram(Img2_cropped);
title("noisy2.png Strip Histogram");


figure;
subplot(2,3,1); imshow(Img2);
title(" Given Noisy Image(noisy2.png)");

subplot(2,3,2); histogram(Img2);
title("Noisy2 Histogram");




% Size of Given Image (noisy2.png):
M1 = size(Img2,1);     
N1 = size(Img2,2);      

% Fourier Transform of Given Image (noisy2.png):

FT1 = fft2(Img2)/(M1*N1);

% Centered Fourier Transform
Center_FT1 = fftshift(FT1);

% Use the abs function to compute the magnitude of the combined components

% In Fourier transforms, high peaks are so high they hide details. 
% Reduce contrast with the log function

% %Fourier spectrum we want to look at

subplot(2,3,3);imshow(abs(log(1 + Center_FT)),[]);
title("Fourier Spectrum");


% Hocam, I took different solid regions with uniform illunination to check
% the noise. The White Region histogram shows that there is Salt
% noise(Which is illusion due to white color and gaussian noise). I realise 
% this after removing Gaussian noise, but still I provide the Median Filter in Comments
% for further enhancement. 

% Histogram of the bottom black strip and black region in Windows show that
% there is Gausssian noise. Which is spatially independent.

% Hocam, Therefore Check Different Solid Region with Uniform Illumination. 



% Step2: Removal of the Identified Noise

% It is clear from the image and it's histogram that seems to be
% similar to Salt [(Which is illusion due to white color and gaussian noise)]
% and Gaussian noise, which is spatially independent. 


% Midpoint and Median Filters  

% Hocam I took this method of implementation of Midpoint filter from Book
% page 231.

% I removed the noise using Midpoint fitler according to slide # 33 It work
% best for Gaussian Noise. I further enhanced the image by applying the
% Median Filter(in Comments) which is very effective for Impulse Noise(Salt&Pepper)
% according to slide #31 lecture 6.


% Hocam I am using direct functions for Filters and Edge detection(Sobel:slide#65 Lec#3)
% because of YOUR PERMISSION to use it. I can implement it using indirect method.

f1_img2 = ordfilt2(Img2, 1, ones(3,3), 'symmetric');
f2_img2 = ordfilt2(Img2, 9, ones(3,3), 'symmetric');
f_img2 = imlincomb(0.5, f1_img2, 0.5, f2_img2);

%f_img2 = ordfilt2(f_img2,5,ones(3,3));

figure;
imshow(f_img2);
imwrite(f_img2,"recovered2.png")
%title("Please Select a Solid Region with Uniform Illumination :)");


% It is clear from the Edges that the Edges became clear after denoising
% the image.

Img2e =edge(Img2,'sobel');
figure;
imshow(Img2e);
title("Combined-edge Before Denoising noisy2.png");

fe2 =edge(f_img2,'sobel');
figure;
imshow(fe2);
title("Combined-edge After Denoising using Midpoint Filter");



%% Step1: Identification of noise via experiment or visual inspection

% Image 3: "noisy3.png"

Img3 = imread("noisy3.png");
imshow(Img3);


% Hocam, Visually, The image seems to be corrupted by by Salt noise. 

% This is my assumption, now I will investigate the spatial domain and then
% Fourier Spectrum of this image to check wether I am wright or wrong.


% I will take a solid area which is uniformly illuminated. Then, the noise 
% can be estimated by analyzing the histogram of this area. 
 
% I cropped a rectangular part of the image for understanding the histogram
% and image noise by using a Mathworks "drawrectangle()" command. I can
% also do it by the method discussed in Lecture-6 Exercise. I chose the
% Mathswork commad.


figure;
imshow(Img3);
title("Please Select a Solid Region with Uniform Illumination :)");


% Following "drawretangle()" command is taken from Mathworks. 
% https://www.mathworks.com/help/images/ref/drawrectangle.html#mw_275b6703-1df2-4d16-aed6-cdfcf02ad8af
% I check different regions of the image from top and bottom corners. The
% noise(Uniform) is same in every solid area from different region.


% "Position" of the ROI, specified as a 1-by-4 numeric vector of the form [xmin, ymin, width, height].
% xmin and ymin specify the coordinates of the upper left corner of the rectangle. 
% width and height specify the width and height of the rectangle and must be nonnegative.


Img3_rect = drawrectangle();
Location = Img3_rect.Position;

x1 = Location(2);
y1 = Location(1);

x2 = round(x1 + Location(4));
y2 = round(y1 + Location(3));

Img3_cropped = Img3(x1:x2 , y1:y2);


figure;
subplot(1,2,1); imshow(Img3_cropped);
title("noisy3.png Strip for Noise Identification");

subplot(1,2,2); histogram(Img3_cropped);
title("noisy3.png Strip Histogram");


figure;
subplot(2,3,1);imshow(Img3);
title(" Given Noisy Image(noisy3.png)");

subplot(2,3,2);histogram(Img3);
title("Noisy3 Histogram");




% Size of Given Image (noisy3.png):
M3 = size(Img3,1);      
N3 = size(Img3,2);      

% Fourier Transform of Given Image (noisy3.png):

FT3 = fft2(Img3)/(M3*N3);

% Centered Fourier Transform
Center_FT3 = fftshift(FT3);

% Use the abs function to compute the magnitude of the combined components

% In Fourier transforms, high peaks are so high they hide details. 
% Reduce contrast with the log function

% %Fourier spectrum we want to look at

subplot(2,3,3);imshow(abs(log(1 + Center_FT)),[]);
title("Fourier Spectrum");



% Step2: Removal of the Identified Noise

% It is clear from the image and it's histogram that seems to be
% similar to Salt Noise, which is spatially independent 


% Median Filter

% Hocam I took this method of implementation of Median filter from Book
% page 231.

% I removed the noise using Median fitler according to slide # 31 It work
% best for Impulse (Salt & Pepper) Noise.

% Hocam I am using direct functions for Filters and Edge detection(Sobel:slide#65 Lec#3)
% because of YOUR PERMISSION to use it. I can implement it using indirect method.


f_img3 = medfilt2(Img3,[3 3], 'symmetric');
%f_img3 = ordfilt2(Img3,5,ones(3,3));

figure;
imshow(f_img3);
imwrite(f_img3,"recovered3.png")


% It is clear from the Edges that the Edges became clear after denoising
% the image.

Img3e =edge(Img3,'sobel');
figure;
imshow(Img3e);
title("Combined-edge Before Denoising noisy3.png");

fe3 =edge(f_img3,'sobel');
figure;
imshow(fe3);
title("Combined-edge After Denoising using Median Filter");
