Command Line Usage:

Training the Model
Train with ResNet architecture (default)
python cube_trainer.py --mode train --iterations 50 --samples 500
Train with LSTM architecture
python cube_trainer.py --mode train --architecture lstm --iterations 50 --samples 500

Demo Solve
python cube_trainer.py --mode demo

Demonstrate solving at specific level with MCTS
python cube_trainer.py --mode demo --level 3 --mcts

Evaluate Performance
Evaluate at current curriculum level
python cube_trainer.py --mode eval 
Evaluate at specific level with MCTS
python cube_trainer.py --mode eval --level 4 --mcts

Solve Custom Scramble
Provide scramble in command line
python cube_trainer.py --mode custom --scramble "R U R' U'"
Or be prompted for scramble
python cube_trainer.py --mode custom