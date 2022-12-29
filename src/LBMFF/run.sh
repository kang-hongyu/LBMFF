cd ../process
python merge_sim.py ../data/SCMFDD-S/drugsim.csv ../data/SCMFDD-S/data2/drugsim.csv ../data/SCMFDD-S/data3/drugsim.csv 0.4 0.6
python merge_sim.py ../data/SCMFDD-S/dissim.csv ../data/SCMFDD-S/data2/dissim.csv ../data/SCMFDD-S/data3/dissim.csv 0.4 0.6
cd ../LBMFF
python main.py

