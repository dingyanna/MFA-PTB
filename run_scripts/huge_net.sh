cd ..
declare exp="huge_net"
declare datasets=("com-friendster.ungraph.txt") # "com-friendster.ungraph.txt")   # "Amazon0302.txt" 
declare dyn='epi' 
for dataset in $datasets; do
    python3 run.py --h 0.0001 --experiment ${exp} --topology real --dynamics $dyn --data ${dataset} --full_max_iter 100000
done 