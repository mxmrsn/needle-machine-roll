clear all; close all; clc;

set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

load('data/gel_data_normed.mat','gel_data_normed');
data = gel_data_normed;

% shuffle dataset so as to not bias training/validation/test datasets
rand_idx = randperm(length(data));
for ii = 1:length(data)
    data_use{ii} = data{rand_idx(ii)};
end

%% Define net and train
width = 50; depth = 1; x_dim = 8; y_dim = 2;
trc = 1; vc = 1; tc = 1;
tr_sz = round(0.9*length(data_use));
va_sz = round(0.05*length(data_use));
ts_sz = round(0.05*length(data_use));
tr_idx = tr_sz;             % last idx for tr
va_idx = tr_idx + va_sz;    % last idx for va
ts_idx = va_idx + ts_sz;    % last idx for ts
for ii = 1:length(data_use)
    if ii<=tr_idx
        training_data{trc} = data_use{ii};
        trc = trc + 1;
    elseif ii>tr_idx && ii<=va_idx
        validate_data{vc} = data_use{ii};
        vc = vc + 1;
    elseif ii>va_idx
        test_data{tc} = data_use{ii};
        tc = tc + 1;
    end
end

kfold = 5; % number of kfold networks in ensemble
for ii = 1:kfold
    clear train_data valid_data data_shuf
    tr_data = [training_data, validate_data];
    rand_id = randperm(length(tr_data));
    for jj = 1:length(tr_data)
        data_shuf{jj} = tr_data{rand_id(jj)};
    end
    tr_sz = round(0.9*length(data_shuf));
    va_sz = round(0.1*length(data_shuf));
    trc = 1; vac = 1;
    for jj = 1:length(tr_data)
        if jj<=tr_sz; train_data{trc} = data_shuf{jj};
        else; valid_data{vac} = data_shuf{jj};
        end
    end
    ii
    res{ii} = trainNetworkAndEvaluateAxangNet(train_data,valid_data,test_data,width,depth,x_dim,y_dim);
end

%% Assess Position Prediction on Test Dataset
h14 = figure(14); h14.Color = 'w'; h14.Units = 'centimeters'; h14.Position(3:4) = [9 6];
h15 = figure(15); h15.Color = 'w'; h15.Units = 'centimeters'; h15.Position = [35.9833    5.1065    9.6838   16.0602];
% Loop through test data
% for ii = 1:length(test_data)
for ii = 5 % test idx to visualize
    for jj = 1:kfold

        ax = res{jj}.XTest{ii}(4:6,:);

        for kk = 1:length(ax)
            z_proj(kk) = dot(ax(:,kk),[0 0 1]);
        end

        ang_pred{ii,jj} = res{jj}.ang_pred{ii};
        ang_test{ii,jj} = res{jj}.ang_test{ii};
        err{ii,jj} = res{jj}.err{ii};

        R_pred = axang2rotm([ax; ang_pred{ii,jj}]');
        R_test = axang2rotm([ax; ang_test{ii,jj}]');

        q_pred = rotm2quat(R_pred);
        q_test = rotm2quat(R_test);

        q_squared_chord_dist = quatChordalSquaredLoss(q_test,q_pred);

        for kk = 1:length(q_test)
            d_ang(kk) = norm(log(R_test(:,:,kk)*R_pred(:,:,kk)'));
        end

        figure(h14);
        subplot(2,1,1);
        plot(rad2deg(err{ii,jj}),'r','LineWidth',1); hold on; grid on; grid minor;
        ylabel('$e_{\theta}$ (deg)'); xlim([0 length(ang_pred{ii,jj})]);
        title('Estimate Error');
        subplot(2,1,2);
        p1 = plot(abs(rad2deg(ang_test{ii,jj})),'k','LineWidth',2); hold on; grid on; grid minor;
        plot(abs(rad2deg(ang_pred{ii,jj}))); xlim([0 length(ang_pred{ii,jj})]);
        ylabel('$\theta$ (deg)');
        legend({'Ground Truth','Estimate'})
        title('Roll Angle Tracking')
        drawnow();
        
        figure(h15);
        p = res{jj}.XTest{ii}(1:3,:).*75;
%         for zz = 1:2:length(R_test)
        for zz = length(R_test)
            clf;
            plot3(p(1,1:zz),p(2,1:zz),p(3,1:zz),'k--'); hold on; grid on; 
            T_gt = [R_test(:,:,zz), p(:,zz); zeros(1,3) 1];
            T_est = [R_pred(:,:,zz), p(:,zz); zeros(1,3) 1];
            drawCoordFrame(T_gt,5,'kkk');
            drawCoordFrame(T_est,5,'rgb');
            buf = 5;
            xmin = min(p(1,:)); xmax = max(p(1,:));
            ymin = min(p(2,:)); ymax = max(p(2,:));
            zmin = min(p(3,:)); zmax = max(p(3,:));
            xlim([xmin-buf xmax+buf]); ylim([ymin-buf ymax+buf]); zlim([zmin-buf zmax+buf]);
            daspect([1 1 1]); view([117 25]);
            drawnow();
        end

    end
    % compute the mean estimate and covariance
    for kk = 1:length(res{jj}.XTest{ii})
        for nn = 1:kfold
            est(:,kk,nn) = ang_pred{ii,nn}(kk);
            errn(:,kk,nn) = err{ii,nn}(kk);
        end
    end
    
    mean_err = rad2deg(mean(errn,3));
    sig_err = rad2deg(std(errn,[],3));
    
    mean_est = rad2deg(mean(abs(est),3));
    sig_est = rad2deg(std(abs(est),[],3));
    
    figure(h14);
    subplot(2,1,1);
    fill_between_lines = @(X,Y1,Y2,col) fill( [X fliplr(X)],  [Y1 fliplr(Y2)], 0.3*col, 'FaceAlpha', 0.3, 'EdgeColor', col); hold on;
    nsigmas = 2;
    fill_between_lines(1:size(errn,2), mean_err+nsigmas.*sig_err, mean_err-nsigmas.*sig_err,[1 0 0]);
    plot(mean_err,'r','LineWidth',2); grid on;
    
    subplot(2,1,2);
    fill_between_lines = @(X,Y1,Y2,col) fill( [X fliplr(X)],  [Y1 fliplr(Y2)], 0.3*col, 'FaceAlpha', 0.3, 'EdgeColor', col); hold on;
    fill_between_lines(1:size(est,2), mean_est+1.*sig_est, mean_est-1.*sig_est,[0 0 1]);
    p2 = plot(mean_est,'b','LineWidth',2); grid on;

    legend([p1 p2],{'Ground Truth','Estimate'});
%     figure(h14); clf;
end

%% Helper Functions
function res = trainNetworkAndEvaluateAxangNet(training_data,validate_data,test_data,width,depth,x_dim,y_dim)

    tr_X = cell(1,length(training_data));
    tr_Y = cell(1,length(training_data));
    tst_X = cell(1,length(test_data));
    tst_Y = cell(1,length(test_data));
    vld_X = cell(1,length(validate_data));
    vld_Y = cell(1,length(validate_data));
    
    for ii = 1:length(training_data)
        tr_X{ii} = training_data{ii}.X(1:x_dim,:);
        tr_Y{ii} = training_data{ii}.Y(1:y_dim,:);
    end
    for ii = 1:length(validate_data)
        vld_X{ii} = validate_data{ii}.X(1:x_dim,:);
        vld_Y{ii} = validate_data{ii}.Y(1:y_dim,:);
    end
    for ii = 1:length(test_data)
        tst_X{ii} = test_data{ii}.X(1:x_dim,:);
        tst_Y{ii} = test_data{ii}.Y(1:y_dim,:);
    end
    
    %% Define Network  
    inputSize = x_dim;
    outputSize = y_dim;
    numHiddenNodes = width;

    layers = [sequenceInputLayer(inputSize)];
    for ii = 1:depth
        layers = [...
                  layers
%                   gruLayer(numHiddenNodes,'OutputMode','sequence')
                  lstmLayer(numHiddenNodes,'OutputMode','sequence') % add ii lstm + dropout layers
                  dropoutLayer(0.2)];
    end
    layers = [...
              layers
              fullyConnectedLayer(outputSize)
              regressionLayer];
    
    options = trainingOptions(...
                            'adam', ...
                            'MiniBatchSize',256, ...
                            'MaxEpochs',1E3, ...
                            'ExecutionEnvironment','gpu', ...
                            'GradientThreshold',1, ...
                            'InitialLearnRate',1E-2, ...
                            'LearnRateSchedule','piecewise', ...
                            'LearnRateDropPeriod',200, ...
                            'LearnRateDropFactor',0.99, ...
                            'Verbose',1, ...
                            'ValidationData',{vld_X vld_Y}, ...
                            'ValidationPatience',5);
    
    %%
    disp('Training Network!');
    [net,info] = trainNetwork(tr_X',tr_Y',layers,options);
    disp('Training Done.');

    %% Test
    disp('Testing With LSTM Network!');
    n = length(tst_X);
    for ii = 1:n
        XTest{ii} = tst_X{ii};        % Ground Truth Input
        YTest{ii} = tst_Y{ii};        % Ground Truth Output

        net = resetState(net); % reset network for new sequence
        numTimeSteps = length(XTest{ii});
        for jj = 1:numTimeSteps
            [net,YPred{ii}(:,jj)] = predictAndUpdateState(net,XTest{ii}(:,jj),'ExecutionEnvironment','gpu'); % ESTIMATE USING NET
        end
        
        % Predict tip angle theta
        ang_test{ii} = atan2(YTest{ii}(1,:),YTest{ii}(2,:));
        ang_pred{ii} = atan2(YPred{ii}(1,:),YPred{ii}(2,:));
        
        err{ii} = abs(ang_test{ii}) - abs(ang_pred{ii}); % in radians
                
    end
    disp('Testing Done.');

    %% Log data to res
    res.net = net; res.info = info;
    res.err = err; res.ang_test = ang_test; res.ang_pred = ang_pred;
    res.YPred = YPred;
    res.YTest = YTest;
    res.XTest = XTest;
end
function err = quatChordalSquaredLoss(q_gt,q_est)
    % compute the chordal squared distance between two quaternions
    for ii = 1:length(q_gt)
        d1 = norm(q_gt(ii,:)-q_est(ii,:));
        d2 = norm(q_gt(ii,:)+q_est(ii,:));
        q_dist = min([d1,d2]);

        err(ii) = 2*q_dist^2 * (4 - q_dist^2);
    end
end
function drawCoordFrame(T,sc,str)
    R = T(1:3,1:3);
    p = T(1:3,4);
    
    x_axis = R(:,1);
    y_axis = R(:,2);
    z_axis = R(:,3);
    
    quiver3(p(1),p(2),p(3),sc*x_axis(1),sc*x_axis(2),sc*x_axis(3),str(1),'LineWidth',2); hold on; grid on;% draw x
    quiver3(p(1),p(2),p(3),sc*y_axis(1),sc*y_axis(2),sc*y_axis(3),str(2),'LineWidth',2);
    quiver3(p(1),p(2),p(3),sc*z_axis(1),sc*z_axis(2),sc*z_axis(3),str(3),'LineWidth',2);
end
