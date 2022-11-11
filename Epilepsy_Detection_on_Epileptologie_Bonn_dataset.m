%Pouya Iranmanesh

clc
clear all
close all
%normal datas
 Path='G:\university\term6\padide\jalase6\B\*.txt';   
 Files = dir(Path);   
 %fda design 
 %%%%%%%%%%%%%Alpha filter design%%%%%%%%%%%%
 Fs = 174;  % Sampling Frequency

Fstop1 = 7;      % First Stopband Frequency
Fpass1 = 8;      % First Passband Frequency
Fpass2 = 12;     % Second Passband Frequency
Fstop2 = 13;     % Second Stopband Frequency
Dstop1 = 0.001;  % First Stopband Attenuation
Dpass  = 0.1;    % Passband Ripple
Dstop2 = 0.001;  % Second Stopband Attenuation
dens   = 20;     % Density Factor

% Calculate the order from the parameters using FIRPMORD.
[N, Fo, Ao, W] = firpmord([Fstop1 Fpass1 Fpass2 Fstop2]/(Fs/2), [0 1 ...
                          0], [Dstop1 Dpass Dstop2]);

% Calculate the coefficients using the FIRPM function.
b  = firpm(N, Fo, Ao, W, {dens});
Hd1 = dfilt.dffir(b);
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%%%%%%%%%%%%Beta filter design%%%%%%%%%%%%
 Fs = 174;  % Sampling Frequency

Fstop1 = 12;      % First Stopband Frequency
Fpass1 = 13;      % First Passband Frequency
Fpass2 = 24;     % Second Passband Frequency
Fstop2 = 25;     % Second Stopband Frequency
Dstop1 = 0.001;  % First Stopband Attenuation
Dpass  = 0.1;    % Passband Ripple
Dstop2 = 0.001;  % Second Stopband Attenuation
dens   = 20;     % Density Factor

% Calculate the order from the parameters using FIRPMORD.
[N, Fo, Ao, W] = firpmord([Fstop1 Fpass1 Fpass2 Fstop2]/(Fs/2), [0 1 ...
                          0], [Dstop1 Dpass Dstop2]);

% Calculate the coefficients using the FIRPM function.
b  = firpm(N, Fo, Ao, W, {dens});
Hd2 = dfilt.dffir(b);
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%%%%%%%%%%%%gama filter design%%%%%%%%%%%%
 Fs = 174;  % Sampling Frequency

Fstop1 = 26;      % First Stopband Frequency
Fpass1 = 27;      % First Passband Frequency
Fpass2 = 69;     % Second Passband Frequency
Fstop2 = 70;     % Second Stopband Frequency
Dstop1 = 0.001;  % First Stopband Attenuation
Dpass  = 0.1;    % Passband Ripple
Dstop2 = 0.001;  % Second Stopband Attenuation
dens   = 20;     % Density Factor

% Calculate the order from the parameters using FIRPMORD.
[N, Fo, Ao, W] = firpmord([Fstop1 Fpass1 Fpass2 Fstop2]/(Fs/2), [0 1 ...
                          0], [Dstop1 Dpass Dstop2]);

% Calculate the coefficients using the FIRPM function.
b  = firpm(N, Fo, Ao, W, {dens});
Hd3 = dfilt.dffir(b);
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 for i=1:length(Files)  
     fn = [Path(1:end-5 ) Files(i,1).name];  
     x = load(fn);
     alpha=filter(Hd1,x);  
     beta=filter(Hd2,x);  
     gama=filter(Hd3,x); 
     f1 = FMD( alpha );% Returns the median
     f2 = FMD( beta );% Returns the median
     f3 = FMD( gama );% Returns the median
     f4 = FMN( alpha );% Returns the mean
     f5 = FMN( beta );% Returns the mean
     f6 = FMN( gama );% Returns the mean
     f7 = FR( alpha );% Returns the ratio of the lowest frequency to the highest frequency
     f8 = FR( beta );% Returns the ratio of the lowest frequency to the highest frequency
     f9 = FR( gama );% Returns the ratio of the lowest frequency to the highest frequency
     f10 = WL( alpha );% Returns the waveform length for the signal
     f11 = WL( beta );% Returns the waveform length for the signal
     f12 = WL( gama );% Returns the waveform length for the signal
     feature_normal(i,:)=[f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12];  
 end
%  subplot(3,1,1)
%  t=linspace(0,23,4097);
%  plot(t,alpha)
%  title('Alpha Band')
%  subplot(3,1,2)
%  plot(t,beta)
%  title('Beta Band')
%  subplot(3,1,3)
%  plot(t,gama)
%  title('Gama Band')

%abnormal datas
 Path='G:\university\term6\padide\jalase6\B\*.txt';
 Files = dir(Path); 
 for i=1:length(Files)   
     fn = [Path(1:end-5 ) Files(i,1).name]; 
     x = load(fn);   
     alpha=filter(Hd1,x); 
     beta=filter(Hd2,x);   
     gama=filter(Hd3,x);  
     f1 = FMD( alpha );  
     f2 = FMD( beta ); 
     f3 = FMD( gama );  
     f4 = FMN( alpha );
     f5 = FMN( beta );   
     f6 = FMN( gama ); 
     f7 = FR( alpha );  
     f8 = FR( beta ); 
     f9 = FR( gama );  
     f10 = WL( alpha );
     f11 = WL( beta );  
     f12 = WL( gama );  
     feature_abnormal(i,:)=[f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12];
 end
 input=[feature_normal;feature_abnormal];
 output=[zeros(100,1);ones(100,1)];
%  output = repelem([{'normal'}, {'eplectic'}], [100 100])';
%  mix=[input output];
%  mix=cell2mat(mix);
%  
x = input';
t = output';
%nnstart
% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network
hiddenLayerSize = 4;
net = patternnet(hiddenLayerSize, trainFcn);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 5/100;

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y);
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);

% View the Network
view(net)
 
% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
figure, plottrainstate(tr)
%figure, ploterrhist(e)
figure, plotconfusion(t,y)
%figure, plotroc(t,y)

