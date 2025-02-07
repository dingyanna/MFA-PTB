cd ..
declare exp="er_density"

python3 run.py --experiment ${exp} --topology er --dynamics epi
python3 run.py --experiment ${exp} --topology er --dynamics gene
python3 run.py --experiment ${exp} --topology er --dynamics wc
python3 run.py --experiment ${exp} --topology er --dynamics eco
 