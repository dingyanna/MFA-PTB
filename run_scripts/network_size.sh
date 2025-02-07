cd ..
declare exp=network_size
 

 
python3 run.py --experiment ${exp} --n 1000 --topology er --dynamics gene 
python3 run.py --experiment ${exp} --n 1000 --topology er --dynamics epi 
python3 run.py --experiment ${exp} --n 1000 --topology er --dynamics wc
python3 run.py --experiment ${exp} --n 1000 --topology er --dynamics eco  