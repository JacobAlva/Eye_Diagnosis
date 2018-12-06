global results
    if ~isempty(x)
        % Choose a Training Function
        % For a list of all training functions type: help nntrain
        % 'trainlm' is usually fastest.
        % 'trainbr' takes longer but may be better for challenging problems.
        % 'trainscg' uses less memory. NFTOOL falls back to this in low memory situations.
        
       trainFcn = 'trainlm';  % Levenberg-Marquardt
        fcn=get(handles.trainfcn,'string');
        choosefcn=fcn{get(handles.trainfcn,'value')};
        switch choosefcn
            case 'trainlm'
                trainFcn = 'trainlm';
            case 'trainscg'
                trainFcn = 'trainscg';
            case 'trainbr'
                trainFcn = 'trainbr';
        end
        %trainFcn = 'trainlm';
        % Create a Fitting Network
       % hiddenLayerSize = [10];
        hiddenLayerSize=str2num(get(handles.hiddenlayer,'string'));
       % net.IW{1,1} = [1 2];
       
        net = fitnet(hiddenLayerSize,trainFcn);
      % transferfcn=get(handles.transfer,'string');
        %net = newff(minmax(x),hiddenLayerSize,{'logsig' 'purelin'},'traingda');
       % net=newff(x,t,hiddenLayerSize,{'tansig' 'tansig' 'tansig'},trainFcn);
        net = init(net);
        net.layers{1}.initFcn = 'initwb';
        net.inputWeights{1,1}.initFcn = 'rands';
        net.inputWeights{2,1}.initFcn = 'rands';
        net.biases{1}.initFcn = 'rands';
        net.biases{2}.initFcn = 'rands';
        % Choose Input and Output Pre/Post-Processing Functions
        % For a list of all processing functions type: help nnprocess
       % net.input.processFcns = {'removeconstantrows','mapminmax','fixunknowns','mapstd','processpca','removerows'};
        net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
        net.outputs{1}.processFcns = {'removeconstantrows','mapminmax'};

        % Setup Division of Data for Training, Validation, Testing
        % For a list of all data division functions type: help nndivide
        net.divideFcn = 'dividerand';  % Divide data randomly
        net.divideMode = 'sample';  % Divide up every sample
        net.divideParam.trainRatio = 70/100;
        net.divideParam.valRatio = 15/100;
        net.divideParam.testRatio = 15/100;

        % Choose a Performance Function
        % For a list of all performance functions type: help nnperformance
        
       % perf=get(handles.performfn,'string');
        perf='mse'
        net.performFcn = perf;  % Mean squared error

        % Choose Plot Functions
        % For a list of all plot functions type: help nnplot
        net.plotFcns = {};
         net.plotFcns = {'plotperform','plottrainstate','ploterrhist','plotregression', 'plotfit','plotconfusion','plotroc'};

        net.trainParam.showWindow=true;
        epc=str2double(get(handles.epoch,'string'));
       % epc=500
        net.trainParam.epochs=epc;
        
        % Train the Network
        
        [net,tr] = train(net,x,t);
       
       

        % Test the Network
        y = net(x);
        e = gsubtract(t,y);
        E = perform(net,t,y);
        
      %   view(net)
%          figure, plotperform(tr)
%         figure, plottrainstate(tr)
%         figure, ploterrhist(e)
%         figure, plotregression(t,y) 
 %       figure, plotfit(net,x,t)
 %       figure, plotconfusion(t,y)
       %figure, plotroc(t,y)
        
    else        
        
        y=inf(size(t));
        e=inf(size(t));
        E=inf;
        
        tr.trainInd=[];
        tr.valInd=[];
        tr.testInd=[];
        
    end

    % All Data
    Data.x=x;
    Data.t=t;
    Data.y=y;
    Data.e=e;
    Data.E=E;
    
    % Train Data
    TrainData.x=x(:,tr.trainInd);
    TrainData.t=t(:,tr.trainInd);
    TrainData.y=y(:,tr.trainInd);
    TrainData.e=e(:,tr.trainInd);
    if ~isempty(x)
        TrainData.E=perform(net,TrainData.t,TrainData.y);
    else
        TrainData.E=inf;
    end
    
    % Validation and Test Data
    TestData.x=x(:,[tr.testInd tr.valInd]);
    TestData.t=t(:,[tr.testInd tr.valInd]);
    TestData.y=y(:,[tr.testInd tr.valInd]);
    TestData.e=e(:,[tr.testInd tr.valInd]);
    if ~isempty(x)
        TestData.E=perform(net,TestData.t,TestData.y);
    else
        TestData.E=inf;
    end
    
    % Export Results
    if ~isempty(x)
        results.net=net;
    else
        results.net=[];
    end
    
    results.Data=Data;
    results.TrainData=TrainData;
    % results.ValidationData=ValidationData;
    results.TestData=TestData;
    