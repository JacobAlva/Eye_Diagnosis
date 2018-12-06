function seg=thresholdtest(I)
handles=guidata(gca);
sum=0;
% dim1=str2double(get(handles.dim1,'string'));
% dim2=str2double(get(handles.dim2,'string'));
[height width]=size(I);
%info=imfinfo(filepath);
axes(handles.axes1)
imshow(I);
for i=1:height
    for j =1:width
        sum=sum+I(i,j);
    end;
end;
avg=sum/(height*width);
tavg=0;
aavg=0;
while(aavg==0)
    sum1=0;sum2=0;
  for i=1:height
    for j =1:width
        if I(i,j)<avg;
            sum1=sum1+I(i,j);
        else sum2=sum2+I(i,j);
        end;
    end;
end;
tavg=(sum1+sum2)/2;
if(tavg<=(avg+20)|| tavg>=(tavg-20));
    aavg=tavg;
else
    avg=tavg;
end;
end;
for i=1:height
    for j =1:width
        if I(i,j)<aavg;
            I(i,j)=0;
        else
            I(i,j)=255;
        end;
    end;
end;
seg=I;
axes(handles.axes1)
imshow(seg);
