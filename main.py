from utils import *

# Options for the RPS Game
state_space = {0: 'rock', 1: 'paper', 2: 'scissors'}
values_list = list(state_space.values())


# Trial Round
def trial_round():
    sequence_moves = []
    # Generate random range for trial round
    x = np.random.randint(6, 9)
    while x > 0:
        player_choice = input("Enter your move (rock, paper, scissors [case sensitive]): ")
        if player_choice not in values_list:
            print("Invalid move. Try again.")
            continue
        player_move = values_list.index(player_choice)

        # Simulate game
        # Check for 'Always Rock' issue
        if len(sequence_moves) >= 3:
            if always_rock_check(sequence_moves=sequence_moves, threshold=3):
                computer_move = (values_list.index(state_space[sequence_moves[-1]]) + 1) % 3
            else:
                computer_move = predict_next_move(state_space=state_space)
        else:
            computer_move = predict_next_move(state_space=state_space)
        # Game Results Logic
        result = determine_winner(player_move=player_move, computer_move=computer_move, seq_moves=sequence_moves)
        print_winner(result, player_move, computer_move, state_space)

        x -= 1
    return sequence_moves


# Main Game
def main():
    player_lives = 3
    game_sequence_moves = trial_round().copy()
    markov_chain = MarkovProgression(sequence_moves=game_sequence_moves)
    # Randomly guess the player's first move
    player_last_move = np.random.choice([0, 1, 2])
    score = 0
    steps = 1
    print(F'GAME BEGINS NOW - YOU HAVE {player_lives} LIVES')
    while player_lives > 0:
        player_choice = input("Enter your move (rock, paper, scissors [case sensitive]): ")
        if player_choice not in values_list:
            print("Invalid move. Try again.")
            continue
        player_current_move = values_list.index(player_choice)

        # Simulate game
        # Check for 'Always Rock' issue
        if steps >= 3:
            # "threshold" signifies the level of tolerance for a repeated sequence of a same move Ex: rock,
            # rock, rock
            if always_rock_check(sequence_moves=game_sequence_moves, threshold=3):
                computer_move = (values_list.index(state_space[game_sequence_moves[-1]]) + 1) % 3
            else:
                computer_move = predict_next_move(markov_instance=markov_chain, prev_move=player_last_move,
                                                  randomness=False)
        else:
            computer_move = predict_next_move(markov_instance=markov_chain, prev_move=player_last_move,
                                              randomness=False)
        # Determine the Winner
        winner = determine_winner(player_move=player_current_move, computer_move=computer_move,
                                  seq_moves=game_sequence_moves)
        # Simple function to print the outputs of each round based on the "winner"
        print_winner(winner, player_current_move, computer_move, state_space)

        # Update Transition Matrix every step:
        markov_chain.update_TMat(last_move=player_last_move, current_move=player_current_move)

        # Update Lives Left and Score based on Winner
        if winner == 'Computer':
            print(f'Lives Left: {player_lives - 1}')
            player_lives -= 1
        elif winner == 'Player':
            score += 1
        player_last_move = player_current_move
        steps += 1

    print(f'Your Total Score: {score}')
    print(markov_chain.transition_matrix)
    print(markov_chain.prob_matrix(), markov_chain.choose_next_move(prev_move=player_last_move))


if __name__ == '__main__':
    main()
