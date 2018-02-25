outputsFile = 'Testingtest.csv';
labelsFile = 'labels_20.csv';

%defining my colors
f1=[0 0 139]/255;
f4=[50 205 50]/255;
f9=[236 0 0]/255;
f14=[85 26 139]/255;
%load the outputs of the model
classResults = csvread(outputsFile);

%load the labels csv
trueLabels = csvread(labelsFile);

classResults = classResults';
[vals indices] = max(classResults);

trueLabels = trueLabels' + 1;

targets = zeros(20,1000);
outputs = zeros(20,1000);

targetsIdx = sub2ind(size(targets), indices, 1:length(indices)); 
outputsIdx = sub2ind(size(outputs), trueLabels, 1:length(trueLabels));

targets(targetsIdx) = 1;
outputs(outputsIdx) = 1;

plotconfusion(targets,outputs)
set(findobj(gca,'color',[0,102,0]./255),'color',f4)
set(findobj(gca,'color',[102,0,0]./255),'color',f9)
set(findobj(gcf,'facecolor',[120,230,180]./255),'facecolor',f4)
set(findobj(gcf,'facecolor',[230,140,140]./255),'facecolor',f9)
set(findobj(gcf,'facecolor',[0.5,0.5,0.5]),'facecolor',f1)
set(findobj(gcf,'facecolor',[120,150,230]./255),'facecolor',f14)