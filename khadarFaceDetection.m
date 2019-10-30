% David Meyer Face Detection 10/22/2019
clc
clear all
close all

%NOTES
%All images 100X100, so already normalized.
%Only have to find the characteristics using eigenvectors. 
%IFF AA' = 0, then same characteristics meaning same person.

%No ID28_###.bmp .   SO... Only 43 people

%% Enhancement
im_pro =imread('Dataset/testing/ID01_010.bmp');%Could encapsulate imread section, it works
im_en=histeq(im_pro); %GrayScale, no black. Brighten it up!
%figure,montage({im_pro,im_en}) %Visual representation for me to see what's
                                %going on.
%% Training
%Lets get the training set
trainList=dir('Dataset/enrolling/*.bmp');%This is really cool that you can make a list this way
im = imread(['Dataset/enrolling/',trainList(1).name]);
[r,c]=size(im);
numOfImages=length(trainList);
numOfPeople=numOfImages/5;%Divide by 5 because they're are 5 pictures of each person

%eigen vector setup...
x=zeros(r*c,numOfPeople);%This is incorrect
vectorOfPeps=zeros(r*c,numOfImages);%This is the eigenvector setup.

Mec=zeros(r*c,1);%Average face, this is frakeinstein, fear the amazing personality.
index=zeros;
index2=zeros;

match=zeros(1,10);%What is this for?
match2=zeros(1,10);
cmc=zeros(1,10);
cmc2=zeros(1,10);

%% Convert to vectors
%%%%%% convert all images to vector %%%%%%
for i=1:numOfImages
    im =imread(['Dataset/enrolling/',trainList(i).name]);
    vectorOfPeps(:,i)=reshape(im',r*c,1); % Has all the image info
end
%% Get Xi and Me
j=1;
for i=1:2:(numOfImages-1)%Change the number 2 here to how many people/pictures
    x(:,j)=(vectorOfPeps(:,i)+vectorOfPeps(:,i+1))./5;%Mean Picture
    Mec(:,1)=Mec(:,1)+vectorOfPeps(:,i)+vectorOfPeps(:,i+1);%Mean Vector
    j=j+1;
end

Me = Mec(:,1) ./ numOfImages;% The different people

%% Get big A

for i=1:numOfPeople
    a(:,i)=x(:,i) - Me;  %Average of person i
end

%% Change to A to P2 for easier computations for the computer
ata = a'*a;  
[V D] = eig(ata);%eig = eigenvectors   The diagonal of the matrix RowEchilonForm
    %V should have the same first column, but it doesn't. Why?
p2 = [];
for i = 1 : size(V,2) 
    if( D(i,i)>1 )
        p2 = [p2 V(:,i)];
    end
end



%%  WEIGHTS
wta=p2'*ata; % A*P2= P;  P'*A =Wt_A



            
%% Get the Eigenfaces    
ef =a*p2;  %here is the P you need to use in matching 
[rr,cc]=size(ef);

for i=1:cc
    eigim_t=ef(:,i);
    eigface(:,:,i)=reshape(eigim_t,r,c);

    %figure,imagesc(eigface(:,:,i)');

    axis image;axis off; colormap(gray(256));
    title('Eigen Face Image','fontsize',10);
end
     
%%
%%%%%%%%%%%%%%%%%%%%%%%  TESTING  %%%%%%%%%%%%%%%%%%%%%%%%
imlist2=dir('Dataset/testing/*.bmp');
numOfImages=length(imlist2);
imt_vector=zeros(r*c,numOfImages);


%%
%%%%%% convert all test images to vector %%%%%%
for i=1:numOfImages
    im =histeq(imread(['Dataset/testing/',imlist2(i).name]));
    imt_vector(:,i)=reshape(im',r*c,1);
    %%%%% get B=y-me %%%%%%%
    b(:,i)=imt_vector(:,i)-Me;  %% bi=imt_vector(i)-Me;
    wtb=ef'*b(:,i);  %%wtb=P'*bi;
    for ii=1:numOfPeople   %% weight compare wtb and wta(i)
        eud(ii)=sqrt(sum((wtb-wta(:,ii)).^5));%Changed from .^2
    end
    [cdata index(i)]=min(eud);  %% find minimum eud's index

       %%%%%%%%%%%%%%%%%%%%%%%  RESULT  %%%%%%%%%%%%%%%%%%%%%%%%
    %%% right result by observation is 1 1 2 3 4 %%%%%
    rresult=[1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 5 5 5 5 5 6 6 6 6 6 7 7 7 7 7 8 8 8 8 8 9 9 9 9 9 10 10 10 10 10 11 11 11 11 11 12 12 12 12 12 13 13 13 13 13 14 14 14 14 14 15 15 15 15 15 16 16 16 16 16 17 17 17 17 17 18 18 18 18 18 19 19 19 19 19 20 20 20 20 20 21 21 21 21 21 22 22 22 22 22 23 23 23 23 23 24 24 24 24 24 25 25 25 25 25 26 26 26 26 26 27 27 27 27 27 29 29 29 29 29 30 30 30 30 30 31 31 31 31 31 32 32 32 32 32 33 33 33 33 33 34 34 34 34 34 35 35 35 35 35 36 36 36 36 36 37 37 37 37 37 38 38 38 38 38 39 39 39 39 39 40 40 40 40 40 41 41 41 41 41 42 42 42 42 42 43 43 43 43 43 44 44 44 44 44];
    %fprintf(rresult)
    %%%%%%%%%%%%%%% CMC calculation %%%%%%%
    %%
    if index(i)==rresult(i)
        match(1)=match(1)+1;%%%%%%%first rank matching number
    else
        [svals,idx]=sort(eud(:));
        index2(i)=idx(2);
        if index2(i)==rresult(i)
            match(2)=match(2)+1;%%%%%%%second rank matching number
        end 
    end
end  
%%
for i=1:10  %% if show CMC of the 1st to 10th rank matching number 
    cmc(i)=sum(match(1:i))/numOfImages;
end
figure,plot(cmc);
title('CMC curve');
            
