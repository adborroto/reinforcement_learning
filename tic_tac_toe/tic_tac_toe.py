import numpy as np

##Tic tac toe environment

class Environment:
    def __init__(self, size=3):
        self.size = size
        self.board = np.zeros((size, size))
        self.winner = None
        self.X = 1
        self.O = -1
        self.EMPTY = 0

    def moves(self):
        m = []
        for i in range(self.size):
            for j in range(self.size):
                if(self.board[i, j] == self.EMPTY):
                    m.append((i, j))
        return m

    def game_over(self):
        
        # Rows
        for p in (self.X, self.O):
            for i in range(self.size):
                if np.sum(self.board[i]) == self.size * p:
                    self.winner = p
                    return True
        # Columns
        for p in (self.X, self.O):
            for i in range(self.size):
                if np.sum(self.board[:, i]) == self.size * p:
                    self.winner = p
                    return True
        # Vertical
        for p in (self.X, self.O):
            if np.trace(self.board) == self.size * p:
                self.winner = p
                return True
            if np.fliplr(self.board).trace() == self.size * p:
                self.winner = p
                return True
        # Full board
        if np.all((self.board != 0)):
            self.winner = None
            return True

        self.winner = None
        return False

    def empty(self, r, c):
        return self.board[r, c] == 0

    def get_reward(self, player):
        if self.winner == player:
            return 1
        if self.winner == None:
            return 0.5
        return -1

    def get_hash_state(self):
        i = 0
        v = 0
        def convert(x): 
            if x == 1:
                return 1
            if x == -1:
                return 2
            return 0
            
        for r in range(self.size):
            for c in range(self.size):
                v += self.size ** i * convert(self.board[r, c])
                i += 1
        return v

    def draw(self):
        for i in range(self.size):
            print("---------------")
            for j in range(self.size):
                print("  ", end="")
                if self.board[i, j] == self.X:
                    print("x |", end="")
                elif self.board[i, j] == self.O:
                    print("o |", end="")
                else:
                    print("  |", end="")
            print("")
        print("---------------")


class Agent:

    def __init__(self, color,verbose=False):
        self.color = color
        self.history = []
        self.V = {}
        self.learning_rate = 0.25
        self.e = 0.25
        self.verbose = verbose

    def get_v(self, state):
        if not state in self.V:
            self.V[state] = 0
        return self.V[state]

    def take_action(self, env):
        moves = env.moves()
        best_action_value = -1
        best_move = None
        pos2value = {} #Debug
        # Epsilon greedy
        r = np.random.random()
        if r < self.e:
            i = np.random.choice(len(moves))
            best_move = moves[i]
        else:
            for i in range(len(moves)):
                move = moves[i]

                env.board[move[0], move[1]] = self.color
                state = env.get_hash_state()
                pos2value[(move[0], move[1])] = self.get_v(state)
                env.board[move[0], move[1]] = 0

                if self.get_v(state) > best_action_value:
                    best_action_value = self.V[state]
                    best_move = move

        env.board[best_move[0], best_move[1]] = self.color
        if self.verbose:
            for i in range(env.size):
                print("------------------")
                for j in range(env.size):
                    if env.board[i,j] == 0:
                        if (i,j) in pos2value:
                            print(" %.2f|" % pos2value[(i,j)], end="")
                        else: 
                            print("?  |",end="")
                    else:
                        print("  ", end="")
                        if env.board[i,j] == env.X:
                            print("x  |", end="")
                        elif env.board[i,j] == env.O:
                            print("o  |", end="")
                        else:
                            print("   |", end="")
                print("")
            print("------------------")

    def update(self, env):
        reward = env.get_reward(self.color)
        target = reward
        last_state = True
        for Vp in reversed(self.history):
            Vprev = 1 if target == 1 and last_state else self.get_v(Vp)
            v = Vprev + self.learning_rate * (target - Vprev)
            self.V[Vp] = v
            target = v
            last_state = False

        self.history = []

    def add_history(self, state):
        self.history.append(state)

class Human:
  def __init__(self, color):
    self.color = color

  def take_action(self, env):
    while True:
      # break if we make a legal move
      move = input("Enter coordinates i,j for your next move (i,j=0..2): ")
      i, j = move.split(',')
      i = int(i)
      j = int(j)
      if env.board[i, j] == 0:
        env.board[i, j] = self.color
        break

  def update(self, env):
      pass

  def add_history(self, state):
      pass

class RandomAgent:
    def __init__(self, color):
        self.color = color
    
    def take_action(self, env):
        moves = env.moves()
        
        i = np.random.choice(len(moves))
        next_move = moves[i]
        env.board[next_move[0], next_move[1]] = self.color


    def update(self, env):
        pass

    def add_history(self, state):
        pass


def change_player(c, p1, p2):
    if c == p1:
        return p2
    else:
        return p1


def play_game(env, p1, p2, verbose=False):
    current_player = None

    while not env.game_over():

        current_player = change_player(current_player, p1, p2)
        current_player.take_action(env)
        hash_state = env.get_hash_state()
        current_player.add_history(hash_state)
        if verbose:
            env.draw()

    p1.update(env)
    p2.update(env)


if __name__ == "__main__":
    size = 3
    p1 = Agent(1)
    p2 = Agent(-1)

    for i in range(5000):
        if i%100 == 0:
            print("Number of games play:", i)
        play_game(Environment(size), p1, p2)
    
    print('Level: ', len(p1.V))

    p1.verbose = True
    while True:
        p1.e = 0
        play_game(Environment(size), p1, Human(-1), verbose=True)
        answer = input("Play again? [Y/n]: ")
        if answer and answer.lower()[0] == 'n':
            break
    
    random = RandomAgent(-1)
    stats = {1: 0, -1: 0, None: 0}
    for i in range(100):
        env = Environment(size)
        p1.e = 0
        play_game(env, p1, random)    
        stats[env.winner]+=1
    print(stats)