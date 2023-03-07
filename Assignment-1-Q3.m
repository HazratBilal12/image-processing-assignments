%%
clear all;
clc;

%% Question # 3: Highlight Flowers in the Image

flowers_img = imread('Flower.png');

figure;
imshow(flowers_img)
title('Image of Flowers');

%% Padding: Applying Zeros around the given Image for Convolution


[M, N] = size(flowers_img); 

Padded_img = uint8(zeros(M+2,N+2));

for i=2:size(Padded_img,1)-1
    for j=2:size(Padded_img,2)-1
        Padded_img(i,j) = flowers_img(i-1,j-1);
    end
end
    

figure; imshow(Padded_img)
title('Zero Padded Image of "Flower.png"');

%% The Max Filter is applied to emphasise the bright details of the image 'Flower.png'


Output_img = Padded_img;
sto_NPV = ones(9,1);     % This varaible is used to store the values of neighbour 
                        % pixels of 3*3, then I applied the max operation
                        % to get the new value and assign to the
                        % 'Output_img'



for i=2:M-1
    for j=2:N-1
        sto_NPV(1) = Padded_img(i-1,j-1);
        sto_NPV(2) = Padded_img(i-1,j);
        sto_NPV(3) = Padded_img(i-1,j+1);
        sto_NPV(4) = Padded_img(i,j-1);
        sto_NPV(5) = Padded_img(i,j);
        sto_NPV(6) = Padded_img(i,j+1);
        sto_NPV(7) = Padded_img(i+1,j-1);
        sto_NPV(8) = Padded_img(i+1,j);
        sto_NPV(9) = Padded_img(i+1,j+1);

        Output_img(i,j) = max(sto_NPV);
        
    end
end

%%
% We can remove the padded zeros using line of code
Output_img = Output_img(2:end-1,2:end-1);


figure; imshow(Output_img)
title('Brighter Image of "Flower.png"');

% Reconstructed Image of 'Flower.png'
Output_img = uint8(Output_img);
imwrite(Output_img, 'EnhancedFlowers.png')



