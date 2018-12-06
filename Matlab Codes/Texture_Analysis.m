
function varargout = Texture_Analysis(varargin)
% texture_analysis MATLAB code for texture_analysis.fig
%      texture_analysis, by itself, creates a new texture_analysis or raises the existing
%      singleton*.
%
%      H = texture_analysis returns the handle to a new texture_analysis or the handle to
%      the existing singleton*.
%
%      texture_analysis('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in texture_analysis.M with the given input arguments.
%
%      texture_analysis('Property','Value',...) creates a new texture_analysis or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Texture_Analysis_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Texture_Analysis_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help texture_analysis

% Last Modified by GUIDE v2.5 31-Oct-2017 18:42:29

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @Texture_Analysis_OpeningFcn, ...
    'gui_OutputFcn',  @Texture_Analysis_OutputFcn, ...
    'gui_LayoutFcn',  [] , ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before texture_analysis is made visible.
function Texture_Analysis_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to texture_analysis (see VARARGIN)

% Choose default command line output for texture_analysis
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);
movegui(hObject,'center');

% UIWAIT makes texture_analysis wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Texture_Analysis_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.uipanel4,'title','CLASSIFICATION RESULT(S)')
global im im2
[path,user_cance]=imgetfile();
if user_cance
    msgbox(sprintf('Error'),'Error','Error');
    return
end
im=imread(path);
% im=im2double(im); %converts to double
% im2=im; %for backup process :)
axes(handles.axes1);
imshow(im);


    set(handles.pushbutton2,'Enable','on');
    set(handles.uitable1,'Data',[])



% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% global im
%imgray=(im(:,:,1)+im(:,:,2)+im(:,:,2))/3;
im=getimage(handles.axes1);
gray=rgb2gray(im);
axes(handles.axes1);
imshow(gray);




% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%----------->here we want to perform Thresholding
% global im
% I = im;

% get image dimensions: an RGB image has three planes
% reshaping puts the RGB layers next to each other generating
% a two dimensional grayscale image

% % BW = im2bw(X,map,0.4);
% % imshow(BW)
%  r = image(:, :, 3);             % red channel
% % g = image(:, :, 2);             % green channel
% % b = image(:, :, 3);             % blue channel
% % threshold = 185;                % threshold value
% bk=otsu(r);
% imshow(bk)

%set(handles.message,'string','Thresholding in progress...');
    
    I=getimage(handles.axes1);
    seg=thresholdtest(I);
    %set(handles.message,'string','Thresholding Complete');



% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% global im
% A=im;
% 
% A_hsv = rgb2hsv(A);
% A_v = A_hsv(:,:,1);
% A_v = histeq(A_v);
% A_hsv(:,:,3) = A_v;
% A = hsv2rgb(A_hsv);
% imshow(A)

I=getimage(handles.axes1);
hist=histeq(I);
axes(handles.axes1)
imshow(hist);

       

% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global im
image = im;
g=getimage(handles.axes1);
Img = g;
pixel_dist = str2double(get(handles.edit1,'String'));
GLCM = graycomatrix(Img,'Offset',[0 pixel_dist; -pixel_dist pixel_dist; -pixel_dist 0; -pixel_dist -pixel_dist]);%the offset values that specify common angles, given the pixel distance D.
stats = graycoprops(GLCM,{'contrast','correlation','energy','homogeneity'});

Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;
axes(handles.axes1);
imshow(Img);

data = get(handles.uitable1,'Data');
data{1,1} = num2str(Contrast(1));
data{1,2} = num2str(Contrast(2));
data{1,3} = num2str(Contrast(3));
data{1,4} = num2str(Contrast(4));
data{1,5} = num2str(mean(Contrast));
first = num2str(mean(Contrast));

data{2,1} = num2str(Correlation(1));
data{2,2} = num2str(Correlation(2));
data{2,3} = num2str(Correlation(3));
data{2,4} = num2str(Correlation(4));
data{2,5} = num2str(mean(Correlation));
second = num2str(mean(Correlation));

data{3,1} = num2str(Energy(1));
data{3,2} = num2str(Energy(2));
data{3,3} = num2str(Energy(3));
data{3,4} = num2str(Energy(4));
data{3,5} = num2str(mean(Energy));
third = num2str(mean(Energy));

data{4,1} = num2str(Homogeneity(1));
data{4,2} = num2str(Homogeneity(2));
data{4,3} = num2str(Homogeneity(3));
data{4,4} = num2str(Homogeneity(4));
data{4,5} = num2str(mean(Homogeneity));
fourth = num2str(mean(Homogeneity));
fifth = fourth + fourth;
z = num2str((mean(Homogeneity)) + (mean(Energy)) + (mean(Contrast)) + (mean(Correlation)));
set(handles.edit5,'string',num2str(mean(Contrast)));
set(handles.edit6,'string',num2str(mean(Correlation)));
set(handles.edit7,'string',num2str(mean(Energy)));
set(handles.edit8,'string',num2str(mean(Homogeneity)));

set(handles.uitable1,'Data',data,'ForegroundColor',[0 0 0])




% --- Executes during object creation, after setting all properties.
function total_average_CreateFcn(hObject, eventdata, handles)
% hObject    handle to total_average (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
axes(handles.axes1)
cla reset
set(gca,'XTick',[])
set(gca,'YTick',[])

set(handles.pushbutton2,'Enable','off')
set(handles.edit1,'String','1')
set(handles.edit5,'String','sometext')
set(handles.uitable1,'Data',[])
set(handles.uitable2,'data',{''})
set(handles.clcountertest,'string',1)



function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end





% --- Executes on button press in pushbutton8.
function pushbutton8_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global datatable
set(handles.uitable1,'data',{''});
set(handles.uitable2,'data',{''});
set(handles.uipanel4,'title','INPUT FEATURE INTO ANN')
datatable=get(handles.dataset,'string');
% n=str2double(get(handles.class,'string'));
 trainno=str2double(get(handles.trainno,'string'));
k=1;
for ct=1:trainno
    for ea=1:3
filepath=strcat(pwd,'\',datatable,'\',num2str(ct),'_',num2str(ea),'.jpg');
set(handles.message,'string','Loading image...');
img=imread(filepath);
axes(handles.axes1);
imshow(img)
pause(0.5)
I=getimage(handles.axes1);
    gray=rgb2gray(I);
    axes(handles.axes1);
    imshow(gray);
    set(handles.message,'string','Converting Image to Grayscale...');
    filepath=strcat(pwd,'\grayimage\',num2str(ct),'_',num2str(ea),'.jpg');
    imwrite(gray,filepath);
hist=histeq(gray);
axes(handles.axes1);
    imshow(hist);
    set(handles.message,'string','Peforming Histogram Equalization...');
   filepath=strcat(pwd,'\histimage\',num2str(ct),'_',num2str(ea),'.jpg');
   imwrite(hist,filepath);
 %I=getimage(handles.axes1);
    seg=thresholdseg(hist);
   % imshow(thresh);
   filepath=strcat(pwd,'\threshimage\',num2str(ct),'_',num2str(ea),'.jpg');
    set(handles.message,'string','Performing Image Thresholding...');
   imwrite(seg,filepath);


     
global im
image = im;

threshold = 135;                % threshold value
Img = seg;
pixel_dist = str2double(get(handles.edit1,'String'));
GLCM = graycomatrix(Img,'Offset',[0 pixel_dist; -pixel_dist pixel_dist; -pixel_dist 0; -pixel_dist -pixel_dist]);
stats = graycoprops(GLCM,{'contrast','correlation','energy','homogeneity'});

Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;

data = get(handles.uitable1,'Data');
data{1,1} = num2str(Contrast(1));
data{1,2} = num2str(Contrast(2));
data{1,3} = num2str(Contrast(3));
data{1,4} = num2str(Contrast(4));
data{1,5} = num2str(mean(Contrast));
first = num2cell(mean(Contrast));

data{2,1} = num2str(Correlation(1));
data{2,2} = num2str(Correlation(2));
data{2,3} = num2str(Correlation(3));
data{2,4} = num2str(Correlation(4));
data{2,5} = num2str(mean(Correlation));
second = num2cell(mean(Correlation));

data{3,1} = num2str(Energy(1));
data{3,2} = num2str(Energy(2));
data{3,3} = num2str(Energy(3));
data{3,4} = num2str(Energy(4));
data{3,5} = num2str(mean(Energy));
third = num2cell(mean(Energy));

data{4,1} = num2str(Homogeneity(1));
data{4,2} = num2str(Homogeneity(2));
data{4,3} = num2str(Homogeneity(3));
data{4,4} = num2str(Homogeneity(4));
data{4,5} = num2str(mean(Homogeneity));
fourth = num2cell(mean(Homogeneity));
%fifth = fourth + fourth;
z = num2str((mean(Homogeneity)) + (mean(Energy)) + (mean(Contrast)) + (mean(Correlation)));
set(handles.edit5,'string',num2str(mean(Contrast)));
set(handles.edit6,'string',num2str(mean(Correlation)));
set(handles.edit7,'string',num2str(mean(Energy)));
set(handles.edit8,'string',num2str(mean(Homogeneity)));

set(handles.uitable1,'Data',data,'ForegroundColor',[0 0 0])

     set(handles.message,'string','GLCM FEATURE EXTRACTION in progress...');
     axes(handles.axes1);
     imshow(Contrast)
    glmcimg=getimage(handles.axes1);
    set(handles.message,'string','FEATURE EXTRACTION Complete');
  
    need=[3:-1:1]
 target=num2cell(need(ea))
  
result=[first second third fourth target]
    cname={'Contrast','Correlation','Energy','Homogeneity','Tag'}
gettable=get(handles.uitable2,'data');
see=gettable(1,1)

if strcmp('',gettable(1,1))==1
set(handles.uitable2,'data',result,'columnname',cname);
else
    gettable=get(handles.uitable2,'data');
    result=[gettable;result]
    set(handles.uitable2,'data',result,'columnname',cname);
end
    


%set(handles.clcounter,'string',k);
k=k+1;
    end
end

table=get(handles.uitable2,'data');
save traintable.mat table

table=cell2mat(table);
[valh indh]=find(table==3);
[vald indd]=find(table==2);
[valg indg]=find(table==1);

health=table(valh,1:4);
t1=[ones(1,numel(valh)) zeros(1,numel(vald)) zeros(1,numel(valg))];
diabetic=table(vald,1:4);
t2=[zeros(1,numel(vald)) ones(1,numel(valh)) zeros(1,numel(valg))];
glaucoma=table(valg,1:4);
t3=[zeros(1,numel(vald)) zeros(1,numel(valg)) ones(1,numel(valh))];

input=[health;diabetic;glaucoma]';
maxnorm=max(max(input,[],2));
input=input./maxnorm;
target=[t1;t2;t3];

x=input;
t=target;

set(handles.message,'string','COMMENCING NEURAL NETWORK TRAINING...');
createandtrainbpnn_module;
pp.results=results;
save(['bpnntrainresult','.mat'],'-struct','pp');
load (['bpnntrainresult','.mat']);
alldata=results.Data.y
trainresult1=results.TrainData.y;
trainresult2=results.TestData.y;

%====Continue===
hidvalue=str2num(get(handles.hiddenlayer,'string'));
S1=hidvalue(1);%5;   % numbe of hidden layers
S2=hidvalue(2);%3;   % number of output layers (= number of classes)


P=input;
T=target;

[R,Q]=size(P); 
epochs = str2double(get(handles.epoch,'string'));%1000;      % number of iterations
goal_err = 10e-5;    % goal error
a=0.3;                        % define the range of random variables
b=-0.3;
W1=a + (b-a) *rand(S1,R);     % Weights between Input and Hidden Neurons
W2=a + (b-a) *rand(S2,S1);    % Weights between Hidden and Output Neurons
b1=a + (b-a) *rand(S1,1);     % Weights between Input and Hidden Neurons
b2=a + (b-a) *rand(S2,1);     % Weights between Hidden and Output Neurons
n1=W1*P;
A1=logsig(n1);
n2=W2*A1;
A2=logsig(n2);
e=A2-T;
error =0.5* mean(mean(e.*e));    
nntwarn off
for  itr =1:epochs
    if error <= goal_err 
        break
    else
         for i=1:Q
            df1=dlogsig(n1,A1(:,i));
            df2=dlogsig(n2,A2(:,i));
            s2 = -2*diag(df2) * e(:,i);			       
            s1 = diag(df1)* W2'* s2;
            W2 = W2-0.1*s2*A1(:,i)';
            b2 = b2-0.1*s2;
            W1 = W1-0.1*s1*P(:,i)';
            b1 = b1-0.1*s1;

            A1(:,i)=logsig(W1*P(:,i),b1);
            A2(:,i)=logsig(W2*A1(:,i),b2);
         end
            e = T - A2;
            error =0.5*mean(mean(e.*e));
            disp(sprintf('Iteration :%5d        mse :%12.6f%',itr,error));
            mse(itr)=error;
    end
end

save W1.mat W1
save W2.mat W2
save A1.mat A1
save A2.mat A2


set(handles.message,'string','NEURAL NETWORK TRAINING COMPLETE');

set(handles.uitable2,'data',{''},'columnname',{''})
msgbox(sprintf('SYSTEM NOW READY TO BE TESTED WITH NEW INPUTS. CLICK ON *TEST SYSTEM...* TO PROCEED'));
    return
% --- Executes on button press in pushbutton9.
function pushbutton9_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton10.
function pushbutton10_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton11.
function pushbutton11_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%set(handles.uitable2,'data',{''})
set(handles.uipanel4,'title','CLASSIFICATION RESULT(S)')

load (['bpnntrainresult','.mat']);
gettable=get(handles.uitable1,'data');
testtable=str2double(gettable(:,4));
alldata=sim(results.net',testtable);

testdata=mean(abs(alldata));

load W1.mat
load W2.mat 
load A1.mat 
load A2.mat

N=testtable./max(testtable);
threshold=0.6;   
% training images result

%TrnOutput=real(A2)
TrnOutput=real(A2>threshold);   

% applying test images to NN
n1=W1*N;
A1=logsig(n1);
n2=W2*A1;
A2test=logsig(n2)

[val ind]=max(A2test,[],1);

%set(handles.clcountertest,'string',1)
ct=str2double(get(handles.clcountertest,'string'));
meanA2test=mean(A2test)

% % testing images result
% 
% %TstOutput=real(A2test)
 TstOutput=real(A2test>threshold);
 
 if ind==3
     result=[num2str(ct) {'Eye has Glaucoma'}]; %num2str(3)%
 else ind==1
     nextval=mean(A2test(1:2,:))
     thresh=0.6
     if nextval>=thresh
         result=[num2str(ct) {'This Eye is Healthy'}];
     else
          result=[num2str(ct) {'Eye is Diabetic'}];
     end
     
 end
 

cname={'S/N','Eye Disease Status'};
gettable=get(handles.uitable2,'data');

%see=gettable(1,1)

if strcmp('',gettable(1,1))==1
set(handles.uitable2,'data',result,'columnname',cname)
else
    gettable=get(handles.uitable2,'data');
    result=[gettable;result];
    set(handles.uitable2,'data',result,'columnname',cname)
end
ct=ct+1;
set(handles.clcountertest,'string',ct)

% --- Executes during object creation, after setting all properties.
function uitable1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to uitable1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes during object creation, after setting all properties.
function text4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called



function edit5_Callback(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit5 as text
%        str2double(get(hObject,'String')) returns contents of edit5 as a double
disp('1');

% --- Executes during object creation, after setting all properties.
function edit5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton14.
function pushbutton14_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function edit6_Callback(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit6 as text
%        str2double(get(hObject,'String')) returns contents of edit6 as a double


% --- Executes during object creation, after setting all properties.
function edit6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit7_Callback(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit7 as text
%        str2double(get(hObject,'String')) returns contents of edit7 as a double


% --- Executes during object creation, after setting all properties.
function edit7_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit8_Callback(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit8 as text
%        str2double(get(hObject,'String')) returns contents of edit8 as a double


% --- Executes during object creation, after setting all properties.
function edit8_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function dataset_Callback(hObject, eventdata, handles)
% hObject    handle to dataset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of dataset as text
%        str2double(get(hObject,'String')) returns contents of dataset as a double


% --- Executes during object creation, after setting all properties.
function dataset_CreateFcn(hObject, eventdata, handles)
% hObject    handle to dataset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton17.
function pushbutton17_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton17 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function trainno_Callback(hObject, eventdata, handles)
% hObject    handle to trainno (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of trainno as text
%        str2double(get(hObject,'String')) returns contents of trainno as a double


% --- Executes during object creation, after setting all properties.
function trainno_CreateFcn(hObject, eventdata, handles)
% hObject    handle to trainno (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function clcountertest_Callback(hObject, eventdata, handles)
% hObject    handle to clcountertest (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of clcountertest as text
%        str2double(get(hObject,'String')) returns contents of clcountertest as a double


% --- Executes during object creation, after setting all properties.
function clcountertest_CreateFcn(hObject, eventdata, handles)
% hObject    handle to clcountertest (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function perffcn_Callback(hObject, eventdata, handles)
% hObject    handle to perffcn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of perffcn as text
%        str2double(get(hObject,'String')) returns contents of perffcn as a double


% --- Executes during object creation, after setting all properties.
function perffcn_CreateFcn(hObject, eventdata, handles)
% hObject    handle to perffcn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function hiddenlayer_Callback(hObject, eventdata, handles)
% hObject    handle to hiddenlayer (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of hiddenlayer as text
%        str2double(get(hObject,'String')) returns contents of hiddenlayer as a double


% --- Executes during object creation, after setting all properties.
function hiddenlayer_CreateFcn(hObject, eventdata, handles)
% hObject    handle to hiddenlayer (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function epoch_Callback(hObject, eventdata, handles)
% hObject    handle to epoch (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of epoch as text
%        str2double(get(hObject,'String')) returns contents of epoch as a double


% --- Executes during object creation, after setting all properties.
function epoch_CreateFcn(hObject, eventdata, handles)
% hObject    handle to epoch (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in trainfcn.
function trainfcn_Callback(hObject, eventdata, handles)
% hObject    handle to trainfcn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns trainfcn contents as cell array
%        contents{get(hObject,'Value')} returns selected item from trainfcn


% --- Executes during object creation, after setting all properties.
function trainfcn_CreateFcn(hObject, eventdata, handles)
% hObject    handle to trainfcn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton8.
function pushbutton18_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function edit21_Callback(hObject, eventdata, handles)
% hObject    handle to perffcn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of perffcn as text
%        str2double(get(hObject,'String')) returns contents of perffcn as a double


% --- Executes during object creation, after setting all properties.
function edit21_CreateFcn(hObject, eventdata, handles)
% hObject    handle to perffcn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit22_Callback(hObject, eventdata, handles)
% hObject    handle to hiddenlayer (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of hiddenlayer as text
%        str2double(get(hObject,'String')) returns contents of hiddenlayer as a double


% --- Executes during object creation, after setting all properties.
function edit22_CreateFcn(hObject, eventdata, handles)
% hObject    handle to hiddenlayer (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit23_Callback(hObject, eventdata, handles)
% hObject    handle to epoch (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of epoch as text
%        str2double(get(hObject,'String')) returns contents of epoch as a double


% --- Executes during object creation, after setting all properties.
function edit23_CreateFcn(hObject, eventdata, handles)
% hObject    handle to epoch (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in trainfcn.
function popupmenu2_Callback(hObject, eventdata, handles)
% hObject    handle to trainfcn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns trainfcn contents as cell array
%        contents{get(hObject,'Value')} returns selected item from trainfcn


% --- Executes during object creation, after setting all properties.
function popupmenu2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to trainfcn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in pushbutton19.
function pushbutton19_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton19 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global datatable
set(handles.uitable1,'data',{''});
set(handles.uitable2,'data',{''});
set(handles.uipanel4,'title','CLASSIFICATION RESULT(S)')
datatable=get(handles.dataset,'string');
% n=str2double(get(handles.class,'string'));
 trainno=str2double(get(handles.trainno,'string'));
k=1;
for ct=1:trainno
    for ea=1
filepath=strcat(pwd,'\',datatable,'\',num2str(ct),'_',num2str(ea),'.jpg');
set(handles.message,'string','Loading image...');
img=imread(filepath);
axes(handles.axes1);
imshow(img)
pause(0.5)
I=getimage(handles.axes1);
    gray=rgb2gray(I);
    axes(handles.axes1);
    imshow(gray);
    set(handles.message,'string','Image Convert to Grayscale...');
    %filepath=strcat(pwd,'\grayimage\',num2str(ct),'_',num2str(ea),'.jpg');
    %imwrite(gray,filepath);
hist=histeq(gray);
axes(handles.axes1);
    imshow(hist);
    set(handles.message,'string','Image Histogram Equalization...');
   % filepath=strcat(pwd,'\histimage\',num2str(ct),'_',num2str(ea),'.jpg');
   % imwrite(hist,filepath);
 %I=getimage(handles.axes1);
    seg=thresholdseg(hist);
    
   % filepath=strcat(pwd,'\threshimage\',num2str(ct),'_',num2str(ea),'.jpg');
    set(handles.message,'string','Image Thresholding...');
   % imwrite(seg,filepath);

%     glcminput=glcmfeature(seg);
     
% global im
% image = im;
% % r = image(:, :, 1);             % red channel
% % g = image(:, :, 1);             % green channel
% % b = image(:, :, 3);             % blue channel
% %threshold = 135;                % threshold value
Img = seg;
pixel_dist = str2double(get(handles.edit1,'String'));
GLCM = graycomatrix(Img,'Offset',[0 pixel_dist; -pixel_dist pixel_dist; -pixel_dist 0; -pixel_dist -pixel_dist]);
stats = graycoprops(GLCM,{'contrast','correlation','energy','homogeneity'});

Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;

data = get(handles.uitable1,'Data');
data{1,1} = num2str(Contrast(1));
data{1,2} = num2str(Contrast(2));
data{1,3} = num2str(Contrast(3));
data{1,4} = num2str(Contrast(4));
data{1,5} = num2str(mean(Contrast));
first = num2cell(mean(Contrast));

data{2,1} = num2str(Correlation(1));
data{2,2} = num2str(Correlation(2));
data{2,3} = num2str(Correlation(3));
data{2,4} = num2str(Correlation(4));
data{2,5} = num2str(mean(Correlation));
second = num2cell(mean(Correlation));

data{3,1} = num2str(Energy(1));
data{3,2} = num2str(Energy(2));
data{3,3} = num2str(Energy(3));
data{3,4} = num2str(Energy(4));
data{3,5} = num2str(mean(Energy));
third = num2cell(mean(Energy));

data{4,1} = num2str(Homogeneity(1));
data{4,2} = num2str(Homogeneity(2));
data{4,3} = num2str(Homogeneity(3));
data{4,4} = num2str(Homogeneity(4));
data{4,5} = num2str(mean(Homogeneity));
fourth = num2cell(mean(Homogeneity));
%fifth = fourth + fourth;
z = num2str((mean(Homogeneity)) + (mean(Energy)) + (mean(Contrast)) + (mean(Correlation)));
set(handles.edit5,'string',num2str(mean(Contrast)));
set(handles.edit6,'string',num2str(mean(Correlation)));
set(handles.edit7,'string',num2str(mean(Energy)));
set(handles.edit8,'string',num2str(mean(Homogeneity)));

set(handles.uitable1,'Data',data,'ForegroundColor',[0 0 0])

     set(handles.message,'string','TESTING FOR HEALTHY IMAGES COMPLETED');
     axes(handles.axes1);
     imshow(Contrast)

   % imshow(thresh);
%     filepath=strcat(pwd,'\glcmimage\',num2str(ct),'_',num2str(ea),'.jpg');
%     imwrite(glmcimg,filepath);
  


load (['bpnntrainresult','.mat']);
gettable=get(handles.uitable1,'data');
testtable=str2double(gettable(:,4));
alldata=sim(results.net',testtable);


testdata=mean(abs(alldata));

load W1.mat
load W2.mat 
load A1.mat 
load A2.mat

N=testtable./max(testtable);
threshold=0.6;   
% training images result

%TrnOutput=real(A2)
TrnOutput=real(A2>threshold);   

% applying test images to NN
n1=W1*N;
A1=logsig(n1);
n2=W2*A1;
A2test=logsig(n2)

[val ind]=max(A2test,[],1);

%set(handles.clcountertest,'string',1)
ct=str2double(get(handles.clcountertest,'string'));
meanA2test=mean(A2test)

% % testing images result
% 
% %TstOutput=real(A2test)
 TstOutput=real(A2test>threshold);
 
 if ind==3
     result=[num2str(ct) {'Eye has Glaucoma'} num2str(3)]; %num2str(3)%
 else ind==1
     nextval=mean(A2test(1:2,:))
     thresh=0.6
     if nextval>=thresh
         result=[num2str(ct) {'This Eye is Healthy'} num2str(1)];
     else
          result=[num2str(ct) {'Eye is Diabetic'} num2str(2)];
     end
     
 end
 

cname={'S/N','Eye Disease Status'};
gettable=get(handles.uitable2,'data');

%see=gettable(1,1)

if strcmp('',gettable(1,1))==1
set(handles.uitable2,'data',result,'columnname',cname)
else
    gettable=get(handles.uitable2,'data');
    result=[gettable;result];
    set(handles.uitable2,'data',result,'columnname',cname)
end
ct=ct+1;
set(handles.clcountertest,'string',ct)

  need=[3:-1:1]
 target=num2cell(need(ea))
  k=k+1;
    end
end


% --- Executes on button press in pushbutton20.
function pushbutton20_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton20 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global datatable
set(handles.uitable1,'data',{''});
set(handles.uitable2,'data',{''});
set(handles.uipanel4,'title','CLASSIFICATION RESULT(S)')
datatable=get(handles.dataset,'string');
% n=str2double(get(handles.class,'string'));
 trainno=str2double(get(handles.trainno,'string'));
k=1;
for ct=1:trainno
    for ea=2
filepath=strcat(pwd,'\',datatable,'\',num2str(ct),'_',num2str(ea),'.jpg');
set(handles.message,'string','Loading image...');
img=imread(filepath);
axes(handles.axes1);
imshow(img)
pause(0.5)
I=getimage(handles.axes1);
    gray=rgb2gray(I);
    axes(handles.axes1);
    imshow(gray);
    set(handles.message,'string','Image Convert to Grayscale...');
    %filepath=strcat(pwd,'\grayimage\',num2str(ct),'_',num2str(ea),'.jpg');
    %imwrite(gray,filepath);
hist=histeq(gray);
axes(handles.axes1);
    imshow(hist);
    set(handles.message,'string','Image Histogram Equalization...');
   % filepath=strcat(pwd,'\histimage\',num2str(ct),'_',num2str(ea),'.jpg');
   % imwrite(hist,filepath);
 %I=getimage(handles.axes1);
    seg=thresholdseg(hist);
    
   % filepath=strcat(pwd,'\threshimage\',num2str(ct),'_',num2str(ea),'.jpg');
    set(handles.message,'string','Image Thresholding...');
   % imwrite(seg,filepath);

%     glcminput=glcmfeature(seg);
     
% global im
% image = im;
% % r = image(:, :, 1);             % red channel
% % g = image(:, :, 1);             % green channel
% % b = image(:, :, 3);             % blue channel
% %threshold = 135;                % threshold value
Img = seg;
pixel_dist = str2double(get(handles.edit1,'String'));
GLCM = graycomatrix(Img,'Offset',[0 pixel_dist; -pixel_dist pixel_dist; -pixel_dist 0; -pixel_dist -pixel_dist]);
stats = graycoprops(GLCM,{'contrast','correlation','energy','homogeneity'});

Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;

data = get(handles.uitable1,'Data');
data{1,1} = num2str(Contrast(1));
data{1,2} = num2str(Contrast(2));
data{1,3} = num2str(Contrast(3));
data{1,4} = num2str(Contrast(4));
data{1,5} = num2str(mean(Contrast));
first = num2cell(mean(Contrast));

data{2,1} = num2str(Correlation(1));
data{2,2} = num2str(Correlation(2));
data{2,3} = num2str(Correlation(3));
data{2,4} = num2str(Correlation(4));
data{2,5} = num2str(mean(Correlation));
second = num2cell(mean(Correlation));

data{3,1} = num2str(Energy(1));
data{3,2} = num2str(Energy(2));
data{3,3} = num2str(Energy(3));
data{3,4} = num2str(Energy(4));
data{3,5} = num2str(mean(Energy));
third = num2cell(mean(Energy));

data{4,1} = num2str(Homogeneity(1));
data{4,2} = num2str(Homogeneity(2));
data{4,3} = num2str(Homogeneity(3));
data{4,4} = num2str(Homogeneity(4));
data{4,5} = num2str(mean(Homogeneity));
fourth = num2cell(mean(Homogeneity));
%fifth = fourth + fourth;
z = num2str((mean(Homogeneity)) + (mean(Energy)) + (mean(Contrast)) + (mean(Correlation)));
set(handles.edit5,'string',num2str(mean(Contrast)));
set(handles.edit6,'string',num2str(mean(Correlation)));
set(handles.edit7,'string',num2str(mean(Energy)));
set(handles.edit8,'string',num2str(mean(Homogeneity)));

set(handles.uitable1,'Data',data,'ForegroundColor',[0 0 0])

     set(handles.message,'string','TESTING FOR DIABETIC IMAGES COMPLETED');
     axes(handles.axes1);
     imshow(Contrast)

   % imshow(thresh);
%     filepath=strcat(pwd,'\glcmimage\',num2str(ct),'_',num2str(ea),'.jpg');
%     imwrite(glmcimg,filepath);
  


load (['bpnntrainresult','.mat']);
gettable=get(handles.uitable1,'data');
testtable=str2double(gettable(:,4));
alldata=sim(results.net',testtable);


testdata=mean(abs(alldata));

load W1.mat
load W2.mat 
load A1.mat 
load A2.mat

N=testtable./max(testtable);
threshold=0.6;   
% training images result

%TrnOutput=real(A2)
TrnOutput=real(A2>threshold);   

% applying test images to NN
n1=W1*N;
A1=logsig(n1);
n2=W2*A1;
A2test=logsig(n2)

[val ind]=max(A2test,[],1);

%set(handles.clcountertest,'string',1)
ct=str2double(get(handles.clcountertest,'string'));
meanA2test=mean(A2test)

% % testing images result
% 
% %TstOutput=real(A2test)
 TstOutput=real(A2test>threshold);
 
 if ind==3
     result=[num2str(ct) {'Eye has Glaucoma'} num2str(3)]; %num2str(3)%
 else ind==1
     nextval=mean(A2test(1:2,:))
     thresh=0.6
     if nextval>=thresh
         result=[num2str(ct) {'This Eye is Healthy'} num2str(1)];
     else
          result=[num2str(ct) {'Eye is Diabetic'} num2str(2)];
     end
     
 end
 

cname={'S/N','Eye Disease Status'};
gettable=get(handles.uitable2,'data');

%see=gettable(1,1)

if strcmp('',gettable(1,1))==1
set(handles.uitable2,'data',result,'columnname',cname)
else
    gettable=get(handles.uitable2,'data');
    result=[gettable;result];
    set(handles.uitable2,'data',result,'columnname',cname)
end
ct=ct+1;
set(handles.clcountertest,'string',ct)

  need=[3:-1:1]
 target=num2cell(need(ea))
  k=k+1;
    end
end

% --- Executes on button press in pushbutton21.
function pushbutton21_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton21 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global datatable
set(handles.uitable1,'data',{''});
set(handles.uitable2,'data',{''});
set(handles.uipanel4,'title','CLASSIFICATION RESULT(S)')
datatable=get(handles.dataset,'string');
% n=str2double(get(handles.class,'string'));
 trainno=str2double(get(handles.trainno,'string'));
k=1;
for ct=1:trainno
    for ea=3
filepath=strcat(pwd,'\',datatable,'\',num2str(ct),'_',num2str(ea),'.jpg');
set(handles.message,'string','Loading image...');
img=imread(filepath);
axes(handles.axes1);
imshow(img)
pause(0.5)
I=getimage(handles.axes1);
    gray=rgb2gray(I);
    axes(handles.axes1);
    imshow(gray);
    set(handles.message,'string','Image Convert to Grayscale...');
    %filepath=strcat(pwd,'\grayimage\',num2str(ct),'_',num2str(ea),'.jpg');
    %imwrite(gray,filepath);
hist=histeq(gray);
axes(handles.axes1);
    imshow(hist);
    set(handles.message,'string','Image Histogram Equalization...');
   % filepath=strcat(pwd,'\histimage\',num2str(ct),'_',num2str(ea),'.jpg');
   % imwrite(hist,filepath);
 %I=getimage(handles.axes1);
    seg=thresholdseg(hist);
    
   % filepath=strcat(pwd,'\threshimage\',num2str(ct),'_',num2str(ea),'.jpg');
    set(handles.message,'string','Image Thresholding...');
   % imwrite(seg,filepath);

%     glcminput=glcmfeature(seg);
     
% global im
% image = im;
% % r = image(:, :, 1);             % red channel
% % g = image(:, :, 1);             % green channel
% % b = image(:, :, 3);             % blue channel
% %threshold = 135;                % threshold value
Img = seg;
pixel_dist = str2double(get(handles.edit1,'String'));
GLCM = graycomatrix(Img,'Offset',[0 pixel_dist; -pixel_dist pixel_dist; -pixel_dist 0; -pixel_dist -pixel_dist]);
stats = graycoprops(GLCM,{'contrast','correlation','energy','homogeneity'});

Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;

data = get(handles.uitable1,'Data');
data{1,1} = num2str(Contrast(1));
data{1,2} = num2str(Contrast(2));
data{1,3} = num2str(Contrast(3));
data{1,4} = num2str(Contrast(4));
data{1,5} = num2str(mean(Contrast));
first = num2cell(mean(Contrast));

data{2,1} = num2str(Correlation(1));
data{2,2} = num2str(Correlation(2));
data{2,3} = num2str(Correlation(3));
data{2,4} = num2str(Correlation(4));
data{2,5} = num2str(mean(Correlation));
second = num2cell(mean(Correlation));

data{3,1} = num2str(Energy(1));
data{3,2} = num2str(Energy(2));
data{3,3} = num2str(Energy(3));
data{3,4} = num2str(Energy(4));
data{3,5} = num2str(mean(Energy));
third = num2cell(mean(Energy));

data{4,1} = num2str(Homogeneity(1));
data{4,2} = num2str(Homogeneity(2));
data{4,3} = num2str(Homogeneity(3));
data{4,4} = num2str(Homogeneity(4));
data{4,5} = num2str(mean(Homogeneity));
fourth = num2cell(mean(Homogeneity));
%fifth = fourth + fourth;
z = num2str((mean(Homogeneity)) + (mean(Energy)) + (mean(Contrast)) + (mean(Correlation)));
set(handles.edit5,'string',num2str(mean(Contrast)));
set(handles.edit6,'string',num2str(mean(Correlation)));
set(handles.edit7,'string',num2str(mean(Energy)));
set(handles.edit8,'string',num2str(mean(Homogeneity)));

set(handles.uitable1,'Data',data,'ForegroundColor',[0 0 0])

     set(handles.message,'string','TESTING FOR GLAUCOMA IMAGES COMPLETED');
     axes(handles.axes1);
     imshow(Contrast)

   % imshow(thresh);
%     filepath=strcat(pwd,'\glcmimage\',num2str(ct),'_',num2str(ea),'.jpg');
%     imwrite(glmcimg,filepath);
  


load (['bpnntrainresult','.mat']);
gettable=get(handles.uitable1,'data');
testtable=str2double(gettable(:,4));
alldata=sim(results.net',testtable);


testdata=mean(abs(alldata));

load W1.mat
load W2.mat 
load A1.mat 
load A2.mat

N=testtable./max(testtable);
threshold=0.6;   
% training images result

%TrnOutput=real(A2)
TrnOutput=real(A2>threshold);   

% applying test images to NN
n1=W1*N;
A1=logsig(n1);
n2=W2*A1;
A2test=logsig(n2)

[val ind]=max(A2test,[],1);

%set(handles.clcountertest,'string',1)
ct=str2double(get(handles.clcountertest,'string'));
meanA2test=mean(A2test)

% % testing images result
% 
% %TstOutput=real(A2test)
 TstOutput=real(A2test>threshold);
 
 if ind==3
     result=[num2str(ct) {'Eye has Glaucoma'} num2str(3)]; %num2str(3)%
 else ind==1
     nextval=mean(A2test(1:2,:))
     thresh=0.6
     if nextval>=thresh
         result=[num2str(ct) {'This Eye is Healthy'} num2str(1)];
     else
          result=[num2str(ct) {'Eye is Diabetic'} num2str(2)];
     end
     
 end
 

cname={'S/N','Eye Disease Status'};
gettable=get(handles.uitable2,'data');

%see=gettable(1,1)

if strcmp('',gettable(1,1))==1
set(handles.uitable2,'data',result,'columnname',cname)
else
    gettable=get(handles.uitable2,'data');
    result=[gettable;result];
    set(handles.uitable2,'data',result,'columnname',cname)
end
ct=ct+1;
set(handles.clcountertest,'string',ct)

  need=[3:-1:1]
 target=num2cell(need(ea))
  k=k+1;
    end
end


% --- Executes on button press in pushbutton22.
function pushbutton22_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton22 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
