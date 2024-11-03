cluster=parcluster();
numWorkers = 144;

% job=batch(cluster,'Data_generation_code_1' ,'Pool',numWorkers-1,'CurrentFolder','/scratch/proj/napamegs/mat_out/Au_n2.5'); %submit job

job=batch(cluster,'Data_generation_code_1' ,'Pool',numWorkers-1,'CurrentFolder','/scratch/proj/napamegs/mat_out/avg_T'); %submit job






