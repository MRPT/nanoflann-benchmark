function [] = analyze_thread_count()
	% Compute the stats from the result files of performance tests wrt
	% the thread count
    close all;
    hold on;

    D=load('THREAD_STATS_DATASET.txt');
    NNPoints = unique(D(:,1));
	NNThread = unique(D(:,2));
        figure(1);

    for i=1:length(NNThread)
        ThreadCount = NNThread(i);
		    idxs = find(D(:,2)==round(ThreadCount));

        for p=1:length(NNPoints)
            NPoint = NNPoints(p);
            points = find(D(idxs,1)==NPoint);
            points=idxs(points);
            TBs = D(points,3); % Time of tree Builds
            TBm   = 1e3*mean(TBs);
            TBms(i,p)=TBm; % Mean time of tree builds
            Speedup(i,p)=TBms(1,p) / TBms(i,p); % Speed-up compared to sequential build
        end
    end
    
    % First plot - build time VS thread count
    for i=1:length(NNThread)
       plot(NNPoints,TBms(i,:), 'DisplayName', sprintf('%d thread', NNThread(i))); 
    end
        
    set(gca,'YGrid','on')
    legend('Location', 'northwest');
    legend show;
    
    ylabel('Tree build time (ms)');
    xlabel('Number of 3D points');
    title('kd-tree build time vs. n.thread.build');
    print("build_time_VS_thread_count.pdf", "-dpdf")
    
    % Clean plot
    clf;
    hold on;
    
    % Second plot - speed up VS thread count
    for i=1:length(NNThread)
       plot(NNPoints,Speedup(i,:), 'DisplayName', sprintf('%d thread', NNThread(i))); 
    end
    
    set(gca,'XScale','log');
    set(gca,'YGrid','on')
    legend('Location', 'northwest');
    legend show;

    ylabel('Speedup ratio');
    xlabel('Number of 3D points');
    title('speedup ratio vs. n.thread.build');
    print("speed_up_VS_thread_count.pdf", "-dpdf")
end