import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm.auto import tqdm
import sys

def train_transition(filename):
    """
    Returns - dataframe with t, u as index and v as columns containing probability of t, u -> v
    
    """
    with open(filename, encoding="utf8") as f:
        lines = f.readlines()
    
    # Store all set of 3 state transitions
    # eg: {(PREVSTART, START): {state1 : 1, ...}}
    transition_dict = {}
    
    # Store all unique states
    states = set()
    states.add('STOP')
    states.add('PREVSTART')
    states.add('START')
    
    prev_prev_state = 'PREVSTART'
    prev_state = 'START'
        
    for line in lines:
        split_line = line.split()
                
        # Start new sequence
        if len(split_line) < 2:
            if (prev_prev_state, prev_state) not in transition_dict.keys():
                transition_dict[(prev_prev_state, prev_state)] = defaultdict(int)
            transition_dict[(prev_prev_state, prev_state)]['STOP'] += 1
            prev_prev_state = 'PREVSTART'
            prev_state = 'START'

        # Processing the current sequence
        elif len(split_line) == 2:
            curr_state = split_line[1]
            states.add(curr_state)
            if (prev_prev_state, prev_state) not in transition_dict.keys():
                transition_dict[(prev_prev_state, prev_state)] = defaultdict(int)
            transition_dict[(prev_prev_state, prev_state)][curr_state] += 1
            prev_prev_state = prev_state
            prev_state = curr_state
            
    # Convert each count to a probability
    for tu, vs in transition_dict.items():
        count_tu = sum(vs.values())
        for v in vs:
            transition_dict[tu][v] = transition_dict[tu][v]/count_tu
    
    # Convert set of states to list 
    states = list(states)
    
    # Generate all possible pairings of states
    all_state_pairs = []
    for t in states:
        for u in states:
            all_state_pairs.append((t, u))
    
    # Create a numpy matrix to store all the transition probabilities
    np_transition_matrix = np.zeros((len(all_state_pairs), len(states)))
    
    for i in range(len(all_state_pairs)):
        for j in range(len(states)):
            tu = all_state_pairs[i]
            v = states[j]
            if tu in transition_dict.keys():
                np_transition_matrix[i][j] = transition_dict[tu[0], tu[1]][v]
            else:
                np_transition_matrix[i][j] = 0
    
    # Convert into DataFrame for easy indexing
    df_transition_matrix = pd.DataFrame(np_transition_matrix, index = all_state_pairs, columns = states)
    
    return df_transition_matrix, all_state_pairs, states

def train_emission(filename, states_w_start_stop, k = 0.5):
    """
    Returns - dataframe with states as index and word as column, containing the probability of state -> word
    """
    with open(filename, encoding="utf8") as f:
        lines = f.readlines()
    
    # for each state y, keep track of each observation count i.e. count (y -> x)
    # before eg: {state1: {obs1: 1, obs2: 5}, state2: {obs1: 4}}
    emission_dict = {}
    
    # update emission_dict for state with count(y -> x) = 0
    # after eg: {state1: {obs1: 1, obs2: 5}, state2: {obs1: 4, obs2: 0}}
    observations = set()
    states = set()
    
    for line in lines:
        split_line = line.split()
        
        # process only valid lines
        if len(split_line) == 2:
            obs, state = split_line[0], split_line[1]
            states.add(state)
            observations.add(obs)
            
            if state not in emission_dict:
                emission_dict[state] = {}
                
            if obs not in emission_dict[state]:
                emission_dict[state][obs] = 1
            else:
                emission_dict[state][obs] += 1

    for key, value in emission_dict.items():
        for obs in observations:
            if obs not in value:
                emission_dict[key][obs] = 0
    
    # Convert state and observation set to list
    states = list(states)
    observations = list(observations) + ['#UNK#'] # Add the #UNK# word into observations at the end
    
    # Create a numpy matrix to store all the emission probabilities
    np_emission_matrix = np.zeros((len(states_w_start_stop), len(observations)))
    for i in range(len(states_w_start_stop)):
        state = states_w_start_stop[i]
        
        if state == 'PREVSTART' or state == 'START' or state == 'STOP': # These states don't have emissions
            continue
            
        v_count = sum(emission_dict[state].values())
        for j in range(len(observations) - 1):
            state = states_w_start_stop[i]
            obs = observations[j]
            np_emission_matrix[i][j] = emission_dict[state][obs]/v_count # count(u -> v) / v_count
        
        # Add Laplace smoothing of k = 0.5 for #UNK# words
        np_emission_matrix[i][len(observations) - 1] = k/(v_count + k)
        
    
    # Convert to df for easy indexing
    df_emission_matrix = pd.DataFrame(np_emission_matrix, index = states_w_start_stop, columns = observations)
    
    return df_emission_matrix, observations

def preprocess_sentence(sentence, training_set_words):
    """
    sentence - a list of Strings (words or observations)
    Returns - a list of Strings, where Strings not in training_set_words are replaced by "#UNK#"
    """
    return [ word if word in training_set_words else "#UNK#" for word in sentence ]

# Simple log function to prevent underflow
def log(x):
    x = np.clip(x, 1e-32, None)
    return np.log(x)

# Main viterbi function
def second_order_viterbi(df_emission_matrix, observations, states, df_transition_matrix, all_state_pairs, sentence, is_preprocessed):
        
    proc_sent = sentence
    if not is_preprocessed:
        proc_sent = preprocess_sentence(sentence, observations)
    proc_sent = ['start'] + proc_sent + ['stop']
    
    # Add 2 for START and STOP states
    n = len(proc_sent)
    
    # Pi Table
    P = pd.DataFrame(index = all_state_pairs, columns = range(n)).fillna(-np.inf)
    
    # Backpointer Table
    B = pd.DataFrame(index = all_state_pairs, columns = range(n))
    
    # Initialisation
    P.loc[("PREVSTART", "START"), 0] = 1
    
    # Forward Recursive Step
    for j in range(1, n-1):
        x = proc_sent[j]
            
        a_b = log(df_transition_matrix.multiply(df_emission_matrix[x], axis = 'columns')) # a(t,u,v) * b(v,x)
        pis = a_b.add(P[j-1], axis = 'index').astype('float64') # a(t,u,v) * b(v,x) + pi at j-1
        
        for curr_state in states:
            if curr_state == 'STOP':
                continue
            tu = all_state_pairs[np.argmax(pis[curr_state].values)] # Get the highest arg for t, u -> v
            score = np.max(pis[curr_state].values) # Get highest score
            P.loc[[(tu[1], curr_state)], j] = score # Store score in Pi table at (u, v)
            B.loc[[(tu[1], curr_state)], j] = tu[0] # Store t in Backtrace table at (u, v)
    
    # Termination
    j = n-1
    a_b = log(df_transition_matrix['STOP']).add(P[j-1],axis = 'index').astype('float64')
    tu = a_b.idxmax() # t, u -> STOP
    score = a_b.max()
    P.loc[[(tu[1], 'STOP')], j] = score
    B.loc[[(tu[1], 'STOP')], j] = tu[0]
    
    # Backtrack
    u, v = P[n-1].astype('float64').idxmax()
    state_seq = []
    for j in range(n-1,0,-1):
        t = B.loc[[(u, v)],j][0]
        state_seq.append(v)
        u, v = t, u
    state_seq = state_seq[::-1][:-1] # Reverse and remove #STOP#
    return state_seq

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage: python hmm_part5.py EN")
        exit()

    dataset = sys.argv[1]
    if dataset not in ['EN', 'SG', 'CN']:
        print("Second argument must be one of the following: ['EN', 'SG, 'CN]")
        exit()
    
    print(f"Evaluating on {dataset}.")
    
    in_file = f"../dataset/{dataset}/dev.in"
    train_file = f"../dataset/{dataset}/train"
    out_file = f"../dataset/{dataset}/dev.p5.out"
    
   # Train
    df_transition_matrix, all_state_pairs, states = train_transition(train_file)
    df_emission_matrix, observations= train_emission(train_file, states)
    
    # Create file handler to write to /dev.p3.out
    outf_h = open(out_file, "w", encoding="utf8")
    
    # Read in file
    with open(in_file, encoding="utf8") as f:
        lines = f.readlines()
        
    sent = [] # initialise array to store 1 sentence at a time.
    for word in tqdm(lines):
        
        if word != "\n":
            sent.append(word.strip())
            
        # We reached end of sentence - time to predict sentence's sequence of states (aka tags)
        else:
            # preprocess sentence (change unknown words to "#UNK#")
            sent_proc = preprocess_sentence(sent, observations)
            # obtain processed sentence's predicted state seq (list of corresponding predicted states for each word in sent)
            sent_state_sequence = second_order_viterbi(df_emission_matrix, observations, states, df_transition_matrix, all_state_pairs, sent_proc, is_preprocessed=True)

            for word, state in zip(sent, sent_state_sequence):
                outf_h.write(word + ' ' + state)
                outf_h.write("\n") # newline for each word
            outf_h.write("\n") # another newline when end of sentence

            # Reset sentence list
            sent = []
    
    outf_h.close()