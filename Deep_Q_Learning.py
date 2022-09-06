#Description: The script is an implementation of Deep Q-Learning for automating the procedure of solving the maze by the agent. 
#It consists of several functions for finding suitable actions and assigning rewards and finally through an iterative procedure, the model learns how to solve the maze in a fixed number of iterations


#Import libraries
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from tensorflow.keras.optimizers import Adam
import time
import numpy as np
import matplotlib.pyplot as plt


# Function for printing the maze if you need it
def print_maze(maze):
    print("██████████")
    for row in maze:
        print("█", end='')
        for col in row:
            if (col == 0):
                print(' ', end='')
            elif (col == 1):
                print('█', end ='')
            elif (col == 2):
                print('O', end='')
        print("█")
    print("██████████")


# Return a clean copy of the maze
def get_new_maze():
    maze = [[2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 0, 0, 0]]
    return maze


# Define actions
actions = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
action_arr = ['U', 'D', 'L', 'R']


# Set up important variables
# Maze info
start_location = (0, 0)
end_location = (7, 7)
width = 8
height = 8
num_action = 4
# Training parameters
alpha = 0.95
random_factor = 0.5
drop_rate = 0.99
max_iterations = 500
max_moves = 256
test_interval = 5 
# Some data
allowed_states = np.zeros((width * height, 4))
current_expectations = np.zeros((width * height, 4))

# One-hot encoding of state index
def one_hot(state):
    width = 8
    height = 8
    data = np.array([state])
    shape = (data.size, width*height)
    one_hot_encoding = np.zeros(shape)
    nrows = np.arange(data.size)
    one_hot_encoding[nrows, data] = 1
    return one_hot_encoding


# Update the state of the maze and agent
def do_move(maze, current_location, end_location, move_idx):
    new_maze = maze
    y, x = current_location
    direction = actions[action_arr[move_idx]]
    new_x = x + direction[1]
    new_y = y + direction[0]
    
    if (new_x >= 0 and new_x <= width-1 and new_y >= 0 and new_y <= height-1): #agent is in the maze
        if (new_maze[new_y][new_x] == 1):                                      #walls in the maze
            new_location = current_location
            current_reward = -1
            return (new_maze, new_location, current_reward)
        
        else:
            new_maze[y][x] = 0
            new_maze[new_y][new_x] = 2
            new_location = (new_y, new_x)
            # Get reward
            if (new_location == end_location):
                current_reward = 0
            else:
                current_reward = -1
            return (new_maze, new_location, current_reward)
    
    else:                                                                      #agent is going out of the maze
        new_location = current_location
        current_reward = -1
        return (new_maze, new_location, current_reward)


# Given an input vector, return the index of the 1 for one-hot encoding   
def get_position_index(loc, width):
    return loc[0] * width + loc[1]


# Try an iteration of the maze in "exploit" only mode
def try_maze(model, max_moves):
    num_moves = 0
    width = 8
    start_location = (0, 0)
    end_location = (7, 7)
    current_location = start_location
    x = []
    y = []
    x.append(current_location[1] + 0.5) 
    y.append(current_location[0] + 0.5)
    maze = get_new_maze()
    while (current_location != end_location and num_moves < max_moves):
        next_move = np.argmax(model.predict(one_hot(get_position_index(current_location, width))))
                    
        # Update the state
        maze, current_location, _ = do_move(maze, current_location, end_location, next_move)
        x.append(current_location[1] + 0.5)
        y.append(current_location[0] + 0.5)
        num_moves += 1
    return num_moves, x, y

# Build the neural network structure
def build_model(maze):
    model = Sequential()
    model.add(Dense(width*height, input_shape=(width*height,),activation='linear'))
    model.add(Dense(width*height, activation='relu'))
    model.add(Dense(num_action, activation='linear'))
    opt = Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=opt)
    return model

# Get the input and target vectors to fit
def get_model_parameters(state_history, width, height):
    inputs = np.zeros((len(state_history), width*height)) #Nx64
    targets = np.zeros((len(state_history), num_action))  #Nx4

    for i in range(len(state_history)):
        prev_state, action, reward, current_state, end_maze = state_history[i]
        prev_state = one_hot(get_position_index(prev_state, width))
        current_state = one_hot(get_position_index(current_state, width))
        inputs[i] = prev_state
        targets[i] = model.predict(prev_state)
        final_target = np.max(model.predict(current_state))
        if end_maze:
            targets[i, action] = reward
        else:
            targets[i, action] = reward + alpha * final_target
    return inputs, targets


# Construct allowed states
maze = get_new_maze()
for y in range(height):
    for x in range(width):
        pos_idx = get_position_index((y, x), width)
        # Allow wall tiles and hence commenting the 2 lines
        #if (maze[y][x] == 1):
        #    continue

        for action_idx in range(len(action_arr)):
            action = action_arr[action_idx]
            new_x = x + actions[action][1]
            new_y = y + actions[action][0]
            if (new_x < 0 or new_x > width-1 or new_y < 0 or new_y > height-1):
                continue
            if maze[new_y][new_x] == 0 or maze[new_y][new_x] == 2:
                allowed_states[pos_idx][action_idx] = 1
                
                
# Initialize rewards
for idx in range(width * height):
    for i in range(0,num_action):
        current_expectations[idx][i] = np.random.uniform (-1.0, -0.1)
        
        
# Set up everything for rendering
# Some variables for stats
train_move_counts = []
test_move_counts = []
consecutive_good_win = 0


# The training happens here
model = build_model(maze)
for iteration in range(max_iterations):
    
    # Reset important variables
    moves = 0
    previous_location = start_location
    current_location = start_location    #for the first iter
    state_history = []
    maze = get_new_maze()
    # Do a run through the maze
    while (current_location != end_location):
        # Choose next action
        next_move = None
        
        # Explore
        if (np.random.random() < random_factor):
            temp_vec = []
            for idx in range(4):
                if (allowed_states[get_position_index(previous_location, width)][idx] == 1):
                    temp_vec.append(idx)
            next_move = np.random.choice(temp_vec)   
            
        # Exploit
        else:   
            next_move = np.argmax(model.predict(one_hot(get_position_index(previous_location, width))))
            
        # Update the state & get reward
        maze, current_location, current_reward = do_move(maze, previous_location, end_location, next_move)
        moves += 1
        
        if ((current_location == end_location) or (moves > max_moves)):
            set_bool = True
        else:
            set_bool = False
        
        # Update state history
        state_history.append((previous_location, next_move, current_reward, current_location, set_bool))
        
        # If agent takes too long, just end the run
        if (moves > max_moves):
            current_location = end_location
        
        previous_location = current_location #for the next iter
    
    
    # Do the learning  
    inputs, targets = get_model_parameters(state_history, width, height)
    h = model.fit(inputs, targets, epochs=4, batch_size=64, verbose=0)
    random_factor = random_factor * drop_rate
    
    # Store number of moves for plotting
    train_move_counts.append(moves)
    print("Iteration:{}".format(iteration + 1))
    print("Moves:{}".format(moves))
    
    # Test the model
    loss = model.evaluate(inputs, targets, verbose=1)
    print()
    
    if ((iteration+1) % test_interval == 0):
        test_val, test_x, test_y = try_maze(model, max_moves)
        if test_val == max_moves:
            print("TEST FAIL")
            consecutive_good_win = 0
        else:
            print("TEST WIN ({} moves)".format(test_val))
            if (test_val < 18):
                consecutive_good_win += 1
            else:
                consecutive_good_win = 0

        # Store number of moves for plotting
        test_move_counts.append(test_val)
        
    # If we've done well enough three times in a row, stop training
    if (consecutive_good_win >= 3):
        break
        
        
# Visualization - moves per iteration
plt.semilogy(train_move_counts, 'b', linewidth=0.5)
plt.xlabel("Iteration Number", fontsize=14)
plt.ylabel("Number of moves taken", fontsize=14)
plt.title("Training moves", fontsize=16)
plt.show()

plt.semilogy(test_move_counts, 'b', linewidth=0.5)
plt.xlabel("Iteration Number", fontsize=14)
plt.ylabel("Number of moves taken", fontsize=14)
plt.title("Testing moves", fontsize=16)
plt.show()

# Visualization - heatmap
# Reset the storage
x = []
y = []

for state, _,_,_,_ in state_history:
    x.append(state[1] + 0.5)
    y.append(state[0] + 0.5)
        
# These will be needed when we are storing previous state instead of current state
if (len(state_history) < 1000):
    x.append(7.5)
    y.append(7.5)
    
if ((iteration + 1) % 250 == 0):
    xedges = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    yedges = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    heatmap, _, _ = np.histogram2d(x, y, bins=(xedges, yedges))
    extent = [0, 8, 0, 8]
    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.xlabel("Agent x-coordinate", fontsize=20)
    plt.ylabel("Agent y-coordinate", fontsize=20)
    plt.gca().invert_yaxis()
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')
    cb = plt.colorbar(ticks=[0, heatmap.max()])
    cb.ax.set_yticklabels(['Never', 'Often'])
    cb.ax.tick_params(labelsize=16)
    time.sleep(0.25)

