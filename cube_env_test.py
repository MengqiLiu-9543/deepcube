
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from cube_env import CubeEnv, create_scrambled_cube

def simple_demo():
    env = CubeEnv()
    state = env.reset(use_oll_pll=True)
    env.render()
    env.save_image("scrambled_cube.png")
    # Define a simple sequence of moves
    moves = ["R", "U", "R'", "U'", "R", "U", "R'", "U'"]
    print(f"\nExecuting sequence: {' '.join(moves)}")

    # Execute each move and print results
    for action in moves:
        next_state, reward, done, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}, Steps: {info['steps']}")
        if done:
            print("Cube solved!")
            break

    print("\nFinal cube state:")
    env.render()
    env.save_image("final_cube.png")

    return env

if __name__ == "__main__":
    simple_demo()
