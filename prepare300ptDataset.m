% Script for Prepping 300pt dataset for training

close all; clc; clear all;
%%
% (SET) Flags
PLOT = 0;
SAVE_TO_FILE = 1;

filedir = pwd;
load(append(filedir,'/data/pts1to60_v2.mat'),'pts1to60');
load(append(filedir,'/data/pts61to120_v2.mat'),'pts61to120_v2');
load(append(filedir,'/data/pts121to200_v2.mat'),'pts121to200');
load(append(filedir,'/data/pts201to300_v2.mat'),'pts201to300');

gel_data(1:60) = pts1to60;
gel_data(61:120) = pts61to120_v2;
gel_data(121:200) = pts121to200;
gel_data(201:300) = pts201to300;

count = 1;
min_x = 0; max_x = 0;
min_y = 0; max_y = 0;
for ii = 1:length(gel_data)
    if ~isempty(gel_data(ii).pose_n)
        dataset(count) = gel_data(ii);
        max_xl = max(gel_data(ii).pose(1,:));
        min_xl = min(gel_data(ii).pose(1,:));
        min_yl = min(gel_data(ii).pose(2,:));
        max_yl = max(gel_data(ii).pose(2,:));
        max_x = max(max_x, max_xl);
        min_x = min(min_x, min_xl);
        max_y = max(max_y, max_yl);
        min_y = min(min_y, min_yl);
        count = count + 1;
    end
end

max_deflection = max(abs([max_x, min_x, max_y, min_y])); % use this to find good bounds for xy

%% Normalize Experimental Data & define inputs/outputs
bounds = [0 75;
          0 75;
          0 75]; % isometric scaling
n = length(dataset);
dataset = dataset(1:n);
h1 = figure(1); h1.Color = 'w'; h1.Units = 'centimeters'; %h1.Position(3:4) = [9 6];
h2 = figure(2); h2.Color = 'w'; h2.Units = 'centimeters'; %h2.Position(3:4) = [9 6];

for ii = 1:n
    
    j2{ii}  = dataset(ii).act(2,:) - dataset(ii).act(2,1);
    sj2{ii} = sin(j2{ii});
    cj2{ii} = cos(j2{ii});
    quat{ii} = dataset(ii).pose_n(4:7,:)';
    axang  = quat2axang(quat{ii});
    ax{ii} = axang(:,1:3);                  
    theta{ii} = axang(:,4)-axang(1,4); % do this to ensure start at zero
    for jj = 1:length(ax{ii})
        zproj{ii}(jj,:) = dot(ax{ii}(jj,:),[0 0 1]);
        if zproj{ii}(jj,:) < 0
            ax{ii}(jj,:) = -1.*ax{ii}(jj,:); % corrected axis
            theta{ii}(jj) = -theta{ii}(jj);  % corrected angle
        end
        T1{ii}(:,:,jj) = [quat2rotm(quat{ii}(jj,:)) dataset(ii).pose_n(1:3,jj); zeros(1,3) 1];
        T2{ii}(:,:,jj) = [axang2rotm([ax{ii}(jj,:) theta{ii}(jj)]) dataset(ii).pose_n(1:3,jj); zeros(1,3) 1];
        dT = T1{ii}(:,:,jj)'*T2{ii}(:,:,jj);
        ang{ii}(jj) = acosd((trace(dT)-1)/2);
    end
    
    stheta = sin(theta{ii});
    ctheta = cos(theta{ii});

    pos = dataset(ii).pose_n(1:3,:);
    pos_normed = minMaxFeatureScaling(pos,bounds)';
        
    X = [pos_normed(1:end-1,:), ax{ii}(1:end-1,:), sj2{ii}(1:end-1)', cj2{ii}(1:end-1)']';                   % axang-roll net (5DOF)
%     X = [pos_normed(1:end-1,:), sj2{ii}(1:end-1)', cj2{ii}(1:end-1)']';                                    % quat net (3DOF)

    Y = [stheta(2:end), ctheta(2:end)]'; % predict angle (5DOF problem)
%     Y = [quat{ii}(2:end,:)]'; % predict quaternion (3DOF problem)
    
    gel_data_normed{ii}.j2 = j2{ii}';
    gel_data_normed{ii}.x  = dataset(ii).pose_n;
    gel_data_normed{ii}.X = X; % input vector
    gel_data_normed{ii}.Y = Y; % output vector
    
    nzeros = 50;
    Vpos = diag(((0.005/75)^2)*ones(1, 3)); % position
    Vang = diag(((0.005)^2)*ones(1, 1));    % angle
    
    noisy_pos_zeros = mvnrnd(zeros(3, 1), Vpos, nzeros)';
    noisy_cos_oness = mvnrnd(ones(1,1), Vang, nzeros)';
    noisy_sin_zeros = mvnrnd(zeros(1,1), Vang, nzeros)';
    
    % Predict Roll Angle from 5DOF and actuation
    gel_data_augmented{ii}.X = [[noisy_pos_zeros; repmat([0 0 1],[nzeros,1])'; noisy_sin_zeros; noisy_cos_oness], X];
    gel_data_augmented{ii}.Y = [[noisy_sin_zeros; noisy_cos_oness], Y];

    gel_data_augmented{ii}.nzeros = nzeros;
    gel_data_augmented{ii}.x = [[zeros(nzeros, 3), repmat([1 0 0 0], [nzeros, 1])]', dataset(ii).pose_n];
    gel_data_augmented{ii}.j2 = j2{ii}';
    
    if (PLOT == 1) 
        figure(h1); 
        subplot(1,3,1);
        plot3(gel_data_normed{ii}.X(1,:),gel_data_normed{ii}.X(2,:),gel_data_normed{ii}.X(3,:)); hold on; grid on;
        scatter3(gel_data_normed{ii}.X(1,end),gel_data_normed{ii}.X(2,end),gel_data_normed{ii}.X(3,end),5,'k','filled');
        daspect([1 1 1]); view(3); 
        xlabel('$n_x$','Interpreter','Latex');
        ylabel('$n_y$','Interpreter','Latex');
        zlabel('$n_z$','Interpreter','Latex'); view(3);

        subplot(1,3,2);
        plot(atan2d(gel_data_normed{ii}.Y(1,:),gel_data_normed{ii}.Y(2,:)),'LineWidth',2); grid on;
        xlabel('Timestamp');
        ylabel('Roll Angle (deg)');

        subplot(1,3,3);
        plot(atan2d(gel_data_normed{ii}.X(7,:),gel_data_normed{ii}.X(8,:)),'LineWidth',2); grid on;
        xlabel('Timestamp');
        ylabel('Actuation Angle (deg)');

        figure(2);
        plot(atan2d(gel_data_normed{ii}.X(7,:),gel_data_normed{ii}.X(8,:))-atan2d(gel_data_normed{ii}.Y(1,:),gel_data_normed{ii}.Y(2,:)),'Color','r','LineWidth',2); grid on;
        xlabel('Timestamp');
        title('\theta-\alpha')
        ylabel('\theta-\alpha')

    end
end

% save gel_data_normed.mat with v7.3 option for export to python
if (SAVE_TO_FILE == 1)
    save('data/gel_data_normed.mat','gel_data_normed','-mat');
end

%% Helper Functions
function data_normed = minMaxFeatureScaling(data,bounds)
    if nargin == 2
        data_min = bounds(:,1);
        data_max = bounds(:,2);
    else
        data_max = max(data,[],2);
        data_min = min(data,[],2);
    end
    
    X_max = data_max.*ones(size(data));
    X_min = data_min.*ones(size(data));
    
    a = 0;
    b = 1;
    A = a.*ones(size(data));
    B = b.*ones(size(data));
    
    num = (data + X_min) .* (B - A);
    den = X_max - X_min;
    
    data_normed = A + (num ./ den);
end


