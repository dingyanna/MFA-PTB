cd ..
 

python3 run.py --n 1000 --experiment sf_heterogeneity --dynamics eco 
python3 run.py --n 2000 --experiment sf_heterogeneity --dynamics eco 

python3 run.py --n 1000 --experiment sf_heterogeneity --dynamics gene 
python3 run.py --n 2000 --experiment sf_heterogeneity --dynamics gene 

python3 run.py --n 1000 --experiment sf_heterogeneity --dynamics epi 
python3 run.py --n 2000 --experiment sf_heterogeneity --dynamics epi 

python3 run.py --n 1000 --experiment sf_heterogeneity --dynamics wc 
python3 run.py --n 2000 --experiment sf_heterogeneity --dynamics wc