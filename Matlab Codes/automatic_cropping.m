
global image gray hist blwt folder

folder='Project_GUI';
image='diseasetrain';
gray='grayimage';
hist='histimage';
blwt='threshimage';
[s m epath]=mkdir(pwd,gray)
[s1 m1 epath1]=mkdir(pwd,hist)
[s2 m2 epath2]=mkdir(pwd,blwt)
%   w=cd
   cd(image)
file=dir('*.jpg')
 %file=dir([image,'.jpg'])
for f=1:numel(file)
    im=file(f).name;
    loadim=imread(im);
    %cropimage=imcrop(loadim);
    resizeimg=imresize(loadim,[100,100])
    grayscale=rgb2gray(resizeimg);
    cd(w)
    cd(gray);
    imwrite(grayscale,im)
    histimg=histeq(grayscale);
    cd(w)
    cd(hist);
    imwrite(histimg,im)
    bk=im2bw(grayscale);
     cd(w)
    cd(blwt);
    imwrite(bk,im)
     cd(w)
     cd(image);
     %cd(folder)
end
cd(cd)
%pwd(folder)