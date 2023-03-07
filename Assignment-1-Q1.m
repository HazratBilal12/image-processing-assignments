clear all;
clc;
%% Question # 1: Decode the Encoded Text in the Image

Given_img = imread('EncodedText.png');

% Now to display the given image with its pixel distribution and color

figure;
subplot(1,2,1); imshow(Given_img)
title('Given Image(Encoded)');

% Investigating the color histogram of given image 'EncondedText.png' 

subplot(1,2,2); histogram(Given_img, 256)
title('Histogram (Color Distribution)');
xlabel('Gray Level of EncodedText.png');
ylabel('Number of Pixels Per Gray Level');

%% Creating Array for Nk, Pr and Sk
% The Size of these arrays are 256(1D Array)

Nk = zeros(256,1);
Pr = zeros(256,1);
Sk = zeros(256,1);

% The size of the EncodedText.png image
% Height and Width 
[M, N] = size(Given_img);

% The total number of pixels of EncodedText.png image

N_pixel = M*N;

%% h(rk)= nk (This step is used to calculate number of pixels nk for a gray level rk)

for rows=1:M
    for cols=1:N
        gray_levels = Given_img(rows, cols); % Gray level of the EncodedText.png image
        Nk(gray_levels+1) = Nk(gray_levels+1) + 1; % Incrementing the gray levels
    end 
end

%% Normalized the histrogram by dividing at N_pixel(M*N)
% Pk(rk)=nk/M*N

Pr = Nk/N_pixel;

%% Histogram Equalization/Linearization

% Sk = T(rk) = (L - 1)Sum Pr(rj) (J=0,1,...K)

% In this a processed (output) image is obtained by mapping each pixel in  
% the input image with intensity rk into a corresponding pixel with level  
% Sk in the output image

c = 1;                            % C is used as a counter to add the previous value
for i=1:256
    
     value = Pr(i);               %  Value is keeping the previous value for next value 
     
     for j=1:c
        if(j<c)
            value = value + Pr(j);    
        end    
     end 
     
     Sk(i) = round(255 * value);
     c = c + 1;  
  
end

%% This section used to find Ps(Sk)
% I used the groupcounts to find the frequency of each index in Sk. Then I
% grouped and added them according to the frequency. finally I divide it by
% MN to find Ps(Sk).


[freq,index] = groupcounts(Sk);
len_array = length(freq);
Ps = zeros(len_array,1);
t=1;
for i=1:len_array
    value2 = 0;
    for j=1:freq(i)
        value2 = value2 + Nk(t);
        t = t + 1;
    end
    
    Ps(i) = value2 / N_pixel;
    
end

%% I applied the Transformation function to the Given_img(EncodedText.png). 

Output_img = uint8(zeros(M,N));

for i=1:M
    for j=1:N
        Output_img(i,j)=Sk(Given_img(i,j)+1);
    end
end

% Processed Image of EncodedText.png
imwrite(Output_img,"DecodedText.png")

%% Reconstructed Image of 'EncodedText.png' with its histogram

figure;
subplot(1,2,1); imshow(Output_img)
title('Decoded Image of EncodedText.png')
subplot(1,2,2); histogram(Output_img,256)
title('Histogram of Output Image(Decoded)')
ylabel('Pixel Count')
xlabel('Gray level')

% Ploting the Different Steps of Histogram Equalization

figure;
plot(Nk);
title('Number of pixels(Nk) having gray level(rk)')

figure;
plot(Pr);
title('Normalized Histogram(Pr(rk)=Nk/M*N)')

figure;
plot(Sk);
title('Transformation function Sk')

figure;
plot(Ps);
title('Uniform Distribution Ps(sk)')




