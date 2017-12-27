function [Z] = cn540_striatum_model(varargin)

% Grosse-Wentrup, M., & Contreras-Vidal, J. L. (2007).
% The role of the striatum in adaptation learning: a computational model.
% Biological Cybernetics, 96(4), 377-388.

% Notes:
% Training the model takes awhile as I made no attempt to optimize
% training.  If you want to skip training after the first time, the script
% will output the weight matrix, which you can then save and load into the
% model via the call cn540_striatum_model(Z);  You can also input your own
% targets via the next argument

%% Variables
% x(1) - position of shoulder (always the origin [0; 0])
% x(2) - position of elbow
% x(3) - position of wrist
% x(4) - end effector position
% target - target position
% theta(1) - shoulder angle
% theta(2) - elbow angle
% theta(3) - wrist angle
% d - distance from the end effector to target (the direction vector magnitude)
% d_init - initial distance from effector to target
% angle_d - angle from the end effector to target (in radians)
% C - input layer cell activity
% R - agonist/antagonist muscle group activations corresponding to
% theta(1), theta(2), theta(3) i.e R(1) = agonist muscle activation, R(2) =
% antagonist muscle activation
% eps - step size

%%

%Setup Figure properties
close all;
figure; polar(0,100);
rectangle('Curvature', [1 1], 'Position', [-65 -10 70 20])
rectangle('Curvature', [1 1], 'Position', [-42 -12.5 25 25])
rectangle('Curvature', [1 1], 'Position', [-36 8 5 5])
rectangle('Curvature', [1 1], 'Position', [-28 8 5 5])
set(gca, 'NextPlot', 'replacechildren');
set(gcf,'doublebuffer','on');
set(0, ...
    'DefaultAxesBox'           , 'off'      ,   ...
    'DefaultAxesTickDir'       , 'out'      ,   ...
    'DefaultAxesTickLength'    , [.02 .02]  ,   ...
    'DefaultAxesXColor'        , [.3 .3 .3] ,   ...
    'DefaultAxesYColor'        , [.3 .3 .3] ,   ...
    'DefaultAxesLineWidth'     , 0.7        ,   ...
    'DefaultAxesFontSize'      , 14         ,   ...
    'DefaultTextFontSize'      , 16         ,   ...
    'DefaultLineLineWidth'     , 4          ,   ...
    'DefaultFigureColor'       , 'white'    ,   ...
    'DefaultFigureNumberTitle' , 'on'             );

%Miscellaneous setup
numTrials = 20;
X = zeros(10290,6);
x_eff_pos = zeros(numTrials,101,2);

eps = 0.05;

%% Train the model (Babbling Phase)

if isempty(varargin)
    [Z] = train_model;
    targets = [-20, 60;...
                20, 60;...
               -20, 20;...
                20, 20 ];
elseif nargin < 2
    Z = varargin{1};
    targets = [-20, 60;...
                20, 60;...
               -20, 20;...
                20, 20 ];
else
    Z = varargin{1};
    targets = varargin{2};
    
end
figure(1);
numTargets = size(targets,1);

%% Test model
for k = 1:numTrials,
    
    %Specify target
    target = targets(mod(k,numTargets)+1, :);
    
    %Turn off the babbling phase generator flag
    ERG = 0;
    
    %Specify initial joint angles
    theta = [pi/8; pi/2; pi/3];
    
    %Given joint angles, find the position of the joints in Cartesian space
    x = update_pos(theta);
    
    %Plot the arm in its current configuration and target
    plot_pos(x, targets);
    pause(.05);
    
    % Calculate the inital magnitude and angle of direction vector, transforming
    % from Cartesian to spherical coordinates,
    d_init = sqrt( (target(1) - x(4,1))^2 + (target(2) - x(4,2))^2);
    d = d_init;
    angle_d = atan2(target(2) - x(4,2) , target(1) - x(4,1));
    display(['d: ', num2str(d)]);
    ind = 1;
    
    for t = 0:eps:5
        display(['trial time: ', num2str(t)]);
        
        %Caclulate the activity of the cells encoding joint and direction
        %vector based on the current direction vector angle and joint angles
        [C] = input_layer(angle_d, theta, ERG, t);
        
        %Update weights based on the firing of the input layer and return
        [Z, R] = update_weights(Z, ERG, C, X);
        
        %update the position of the joints based on new muscle activations R
        theta = theta + eps * delta(d_init, d) * diff(reshape(R, 2,3))';
        
        %update the new arm position based on the new joint angles
        x = update_pos(theta);
        
        %plot the arm in its current configuration and target
        plot_pos(x, targets);
        
        %update distance vector
        d = sqrt( (target(1) - x(4,1))^2 + (target(2) - x(4,2))^2);
        angle_d = atan2(target(2) - x(4,2) , target(1) - x(4,1));
        display(['d: ', num2str(d)]);
        
        %store the end effector position for later plotting
        x_eff_pos(k, ind,:) = x(4,:);
        ind = ind + 1;
        
        if d < 1
            break;
        end
        
    end
    
end

%plot path of end effector
figure; polar(0,100); hold all;
plot(x_eff_pos(:, :, 1), x_eff_pos(:, :,2), 'r.', 'MarkerSize', 20);
plot(targets(:,1), targets(:,2), 'bo', 'MarkerSize', 20);
title('Path of Reaches');

end


function [out] = delta(d_init, d)

% scaling function ensuring approximately bellshaped velocity curves
% characteristic of reaching movements

%Constants
alpha = 0.9;
beta = 0.002;
gamma = 0.1;

%Ratio of distance from target compared to initial distance from target
dist_ratio = d/d_init;

display(['dist_ratio: ', num2str(dist_ratio)]);

if 0 <= (dist_ratio) && (dist_ratio) <= .5
    out = ((alpha * (2 * dist_ratio - 2)^4) / (beta + (2 * dist_ratio - 2)^4)) + gamma;
elseif dist_ratio <= 1
    out = ((alpha * (2 * dist_ratio)^4) / (beta + (2 * dist_ratio)^4)) + gamma;
else
    out = ((alpha * (2 * dist_ratio)^4) / (beta + (2 * dist_ratio)^4)) + gamma;
    %error('distance ratio out of bounds');
end

end

function [x] = update_pos(theta)
%% Based on the joint angles, figure out the arm position

l1 = 28; % link one length
l2 = 28; % link two length
l3 = 16; % link three length

%% Enforce Constraints on Joint Angle

if theta(1) < (-pi/2)
    theta(1) = (-pi/2)+.1;
elseif theta(1) > pi
    theta(1) = pi - .1;
end

if theta(2) < 0
    theta(2) = .1;
elseif theta(2) > pi
    theta(2) = pi - .1;
end

if theta(3) < (-pi/2)
    theta(3) = (-pi/2) + .1;
elseif theta(3) > (pi/2)
    theta(3) = (pi/2) - .1;
end

%% Update Position
x2 = [l1 * cos(theta(1)); ...
    l1 * sin(theta(1))];
x3 = x2 + [l2*cos(theta(1) + theta(2)); ...
    l2*sin(theta(1) + theta(2))];
x_eff = x3 + [l3*sin((pi/2) - sum(theta)); ...
    l3*cos((pi/2) - sum(theta))];


x = [zeros(2,1), x2, x3, x_eff]';


end

function [C] = input_layer(angle_d, theta, ERG, t)

% Divide the possible direction vector angle space into 30 angular regions
d_maxpref_angles = (-pi+(pi/60):2*pi/30:pi)';

% Divide the possible joint configuration angles into 7 angular regions
joint_maxpref_angles(1,:) = (((-pi/2)+(3*pi/14)):((3*pi/2)/7):pi)';
joint_maxpref_angles(2,:) = (pi/14:pi/7:pi)';
joint_maxpref_angles(3,:) = ((-pi/2)+(pi/14):pi/7:(pi/2))';

% Find all possible combinations of joint and direction vector
% configurations.  This corresponds to 10,290 cells, each of which fire
% maximally when corresponding to one of these configurations of direction
% vector and joint angles
[B{1:4}] = ndgrid(d_maxpref_angles, joint_maxpref_angles(1,:), joint_maxpref_angles(2,:), joint_maxpref_angles(3,:));
maxpref = reshape(cat(4+1,B{:}),[],4) ;

% Calculate input layer activity
C = 4 - (1/pi) * (abs(angle_d - maxpref(:,1)) + ...
    sum(abs(repmat(theta, [1 length(maxpref)])' ...
    - maxpref(:,2:4)),2)); % cell population

% Take only the 7 most active cells, set the rest to zero (for
% computational speed) in the first 50% of trials, else take only the 3
% most active
[temp, top7_ind]=sort(C, 'descend');

if (ERG == 1) && (t < 20000)
    C(~ismember(1:10290, top7_ind(1:7))) = 0;
else
    C(~ismember(1:10290, top7_ind(1:3))) = 0;
end

% Normalize to the most active cell, making sure that there is no division
% by zero
if max(C) ~= 0,
    C = C/max(C);
else
    C = zeros(10290,1);
end

end

function [Z] = train_model()

%Initialize weights at 0
Z = zeros(10290,6);

% Set step size
eps = 0.05;
% Turn on random generator for babbling
ERG = 1;

% Plot weights and cell inputs.  Comment out for faster training
% figure;
% imagesc(Z');
% subplot(212); colorbar; shading interp; colormap(summer); grid off; caxis([0 1]);
% set(gca, 'NextPlot', 'replacechildren');
% set(gcf,'doublebuffer','on');
% figure(2);

% Begin Training
for t = 0:40000
    if mod(t,10) == 0,
        %generate new joint configuration every 10 trials
        theta = rand(3,1); %generate random numbers between [0 1]
        theta(1) = (theta(1) - .5)*2*(pi); %map to between [-pi pi]
        theta(2) = theta(2) * pi; % map to between [0 pi]
        theta(3) = (theta(3) - .5)*pi; %map to between [-pi/2 pi/2]
        
        %Find position of arm based on new joint configuration
        x = update_pos(theta);
        display(['Training Trial: ',num2str(t)]);
    end
    
    % Randomly set one of the antagonist/agonist pairs to active
    
    R = rand(1,6);
    
    if rand(1) < .5
        R(1) = 0;
    else
        R(2) = 0;
    end
    if rand(1) < .5
        R(3) = 0;
    else
        R(4) = 0;
    end
    if rand(1) < .5
        R(5) = 0;
    else
        R(6) = 0;
    end
    for j = 1:.5:5
        % Update joint and arm position based on random arm movement
        old_theta = theta;
        theta = theta + eps * diff(reshape(R, 2,3))';
        old_x_eff = x(4,:);
        x = update_pos(theta);
        
        % The new direction vector points from the old x_eff position to the new
        % x_eff position
        
        % Find angle of direction vector
        angle_d = atan2(x(4,2) - old_x_eff(2), x(4,1) - old_x_eff(1));
        
        % Compute input layer firing based on the new direction vector angle
        % and old joint angles
        [C] = input_layer(angle_d, old_theta, ERG, t);
        
        %     old_Z = Z;
        
        % Update weights based on input layer cell firing
        [Z] = update_weights(Z, ERG, C, R);
        %     display(['weight change statistic:', num2str(sqrt(sum((Z - old_Z).^2)/10290))]);
        %     display(num2str(max(Z)));
    end
    %         Animated Weight matrix and inputs.  Comment out for faster training.
    %         clf;
    %         subplot(212)
    %         imagesc(Z'); colorbar; grid off; box off; shading interp;
    %         xlim([0 10291]);
    %         caxis([0 1]);
    %         title('Weight Matrix');
    %         subplot(211);
    %
    %         bar(1:10290,C);
    %         title('Cell Inputs');
    %         text(2677,.5, ['Training Trial: ', num2str(t)]);
    %         grid off; box off;
    %         ylim([0 1]);
    %         xlim([0 10291]);
    %         pause(.02);
end


end

function [Z, R] = update_weights(Z, ERG, C, X)

% Set constant for agonist/antagonist weight update
eta = 0.4;
% Step Size
dt = .05;

% If we are not training the model, the muscle activations are the inner
% product of the weights and the input layer activities
if ERG ~= 1
    R = C'*Z;
else
    R = X;
end

% display('C: ');
% display(num2str(C(C>0)'));
% display('R: ');
% display(num2str(R));
% display('inner product: ');
% display(num2str(C'*Z));

% Outstar learning weight update
Z = Z + dt*eta*(repmat(C,1,length(R)).*(repmat(R,length(C),1)-Z));

end

function plot_pos(x, targets)
% Plots the position of the arm and the target

clf;
polar(0,100);
hold on;
rectangle('Curvature', [1 1], 'Position', [-65 -10 70 20])
rectangle('Curvature', [1 1], 'Position', [-42 -12.5 25 25])
rectangle('Curvature', [1 1], 'Position', [-36 8 5 5])
rectangle('Curvature', [1 1], 'Position', [-28 8 5 5])

line(x(:,1), x(:,2));
p1 = plot(x(4,1), x(4,2), 'ro', 'MarkerSize', 20);
p2 = plot(targets(:,1), targets(:,2), 'bo', 'MarkerSize', 20);
legend([p1 p2], {'Hand', 'Targets'})
pause(.05);
end