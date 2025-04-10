How to Use:

Training a Model
python cube_trainer.py --mode train --epochs 20 --samples 1000 --use_oll_pll

Testing with a Custom Scramble
python cube_trainer.py --mode solve_custom --custom_scramble "R U R' U'"

Testing with a Random OLL+PLL Scramble
python cube_trainer.py --mode solve_random --use_oll_pll