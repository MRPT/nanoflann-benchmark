clear;

% columns: PC_SIZE BUILD QUERY
R=load('results-nanoflann-kitti.txt');

% Sort by 1-st column:
R=sortrows(R);

afigure;
aplot(R(:,1),1e3*R(:,2),'.');
title('Build time');
xlabel('Point cloud size');
ylabel('kd-tree build time [ms]');

afigure;
aplot(R(:,1),1e6*R(:,3),'k.');
title('Query time');
xlabel('Point cloud size');
ylabel('kd-tree query time [\mu s]');
