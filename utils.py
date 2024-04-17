import numpy as np
import scipy.linalg


# Markov Logic to Decide the Computers Moves Based on Predicted Outcomes
class MarkovProgression:
    def __init__(self, sequence_moves):
        self.transition_matrix = np.zeros(shape=(3, 3))
        self.calc_init_TMat(sequence_moves)

    def calc_init_TMat(self, sequence_moves):
        for i in range(len(sequence_moves) - 1):
            current_move = sequence_moves[i]
            next_move = sequence_moves[i + 1]
            self.transition_matrix[current_move][next_move] += 1

    # Update Transition Matrix
    def update_TMat(self, last_move, current_move):
        self.transition_matrix[last_move][
            current_move] += 1  # adds the new information, ensuring that the latest move has a weighted impact on
        # the matrix.
        return self.transition_matrix

    def prob_matrix(self):
        # Normalize the matrix so that each row sums to 1
        probabilities = self.transition_matrix / self.transition_matrix.sum(axis=1)[:, np.newaxis]
        return probabilities

    def choose_next_move(self, prev_move, randomness=False):
        # Computer's move to beat the player's most likely move
        # Rock (0) beats Scissors (2), Paper (1) beats Rock (0), Scissors (2) beats Paper (1)
        beating_moves = {0: 1, 1: 2, 2: 0}
        probabilities = self.prob_matrix()[prev_move]
        # unique_probs = np.unique(probabilities)
        if randomness and (np.random.rand() < 0.5):  # allow 50% of the moves to be random in nature
            computer_move = np.random.choice([0, 1, 2])
        else:
            # if len(unique_probs) < len(probabilities):  # This means there are duplicates
            #   player_move = np.random.choice([0, 1, 2])
            # else:
            player_move = int(np.argmax(probabilities))
            # Choose the computer's move to beat the player's most likely move
            computer_move = beating_moves[player_move]
        return computer_move

    # Calculate Stationary Distribution
    def stationary_distribution(self):
        prob_mat = self.prob_matrix()
        # Calculate the Eigenvalues and Eigenvectors (left only)
        eigenvalues, left_ev = scipy.linalg.eig(prob_mat, left=True, right=False)
        # Choose the Eigenvectors whose corresponding (Real) Eigenvalue is 1
        stationary_dist = np.abs(left_ev[:, np.argmax(np.isclose(eigenvalues, 1))]).astype(float)
        # Divide the values in each row by the sum of that row
        stationary_dist /= stationary_dist.sum()
        return stationary_dist

    # Compute the Computers Next Choice based on the Stationary Distribution
    def compute_NextMove(self, randomness=False):
        stat_dist = self.stationary_distribution()
        # Computer's move to beat the player's most likely move
        # Rock (0) beats Scissors (2), Paper (1) beats Rock (0), Scissors (2) beats Paper (1)
        beating_moves = {0: 1, 1: 2, 2: 0}
        # Find the player's most likely move based on the stationary distribution
        player_move = int(np.argmax(stat_dist))  # Stationary distribution probabilities for player's moves
        if randomness and (np.random.rand() < 0.5):  # allow 50% of the moves to be random in nature
            computer_move = np.random.choice([0, 1, 2])
        else:
            # Choose the computer's move to beat the player's most likely move
            computer_move = beating_moves[player_move]
        return computer_move


# Basic functions for the game
def determine_winner(player_move, computer_move, seq_moves):
    # Append the player move to the sequence of moves
    seq_moves.append(player_move)
    if player_move == computer_move:
        return 'TIE'
    # Cases where Player Wins
    elif (player_move == 0 and computer_move == 2) or \
            (player_move == 1 and computer_move == 0) or \
            (player_move == 2 and computer_move == 1):
        return 'Player'
    # Cases where Player Loses:
    elif (player_move == 2 and computer_move == 0) or \
            (player_move == 1 and computer_move == 2) or \
            (player_move == 0 and computer_move == 1):
        return 'Computer'


def print_winner(result, player_move, computer_move, state_space):
    if result == 'TIE':
        print(f'Your Move: {state_space[player_move]}, Computers Move: {state_space[computer_move]} Outcome => TIE')

    elif result == 'Player':
        print(
            f'Your Move: {state_space[player_move]}, Computers Move: {state_space[computer_move]} Outcome => You win!')

    elif result == 'Computer':
        print(
            f'Your Move: {state_space[player_move]}, Computers Move: {state_space[computer_move]} Outcome => You lose!')


def predict_next_move(state_space=None, markov_instance=None, prev_move=None, randomness=False):
    if markov_instance is None:
        return np.random.choice(list(state_space.keys()))
    elif state_space is None:
        # Use the MarkovProgression instance to predict the next move
        # return markov_instance.compute_NextMove(randomness=randomness)
        return markov_instance.choose_next_move(prev_move, randomness)


def always_rock_check(sequence_moves, threshold=3):
    """
    Check if the player is consistently choosing one option.
    sequences_moves (list): The sequence of moves made by the player
    threshold (int): The number of the same consecutive choices to trigger a check
    """
    if sequence_moves[-3:].count(sequence_moves[-1]) == threshold:
        return True
