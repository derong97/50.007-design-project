from tqdm.auto import tqdm
import numpy as np
import sys

# Emissions
def train_emission(filename):
    """
    Returns - a dictionary containing emission parameters
    """
    with open(filename, encoding="utf8") as f:
        lines = f.readlines()
    
    # for each state y, keep track of each observation count i.e. count (y -> x)
    # before eg: {state1: {obs1: 1, obs2: 5}, state2: {obs1: 4}}
    emission_dict = {}
    
    # update emission_dict for state with count(y -> x) = 0
    # after eg: {state1: {obs1: 1, obs2: 5}, state2: {obs1: 4, obs2: 0}}
    observations = set()
    
    for line in lines:
        split_line = line.split()
        
        # process only valid lines
        if len(split_line) == 2:
            obs, state = split_line[0], split_line[1]
            
            observations.add(obs)
            
            if state not in emission_dict:
                emission_dict[state] = {}
                
            if obs not in emission_dict[state]:
                emission_dict[state][obs] = 1
            else:
                emission_dict[state][obs] += 1

    for k, v in emission_dict.items():
        for obs in observations:
            if obs not in v:
                emission_dict[k][obs] = 0
    
    return emission_dict

def get_emission_params_fixed(emission_dict, state, obs, k=0.5):
    
    if state not in emission_dict:
        raise Exception("State not in emission dict")
    
    state_data = emission_dict[state]
    count_y = sum(state_data.values()) # count(y)
    
    if obs == "#UNK#":
        count_y_to_x = k
    else:
        count_y_to_x = state_data[obs] # count(y -> x)
    
    return count_y_to_x / (count_y + k)

# Transitions
def train_transition(filename):
    """
    Returns - a dictionary containing transition parameters
    """
    with open(filename, encoding="utf8") as f:
        lines = f.readlines()
    
    # for each state u, keep track of each state count i.e. count (u,v)
    # before eg: {START: {y1: 1, y2: 5}, y1: {y1: 3, y2: 4, STOP: 1}, y2: {y1: 1, STOP: 3}}
    transition_dict = {}
    
    # after eg: {START: {y1: 1, y2: 5, STOP: 0}, y1: {y1: 3, y2: 4, STOP: 1}, y2: {y1: 1, y2: 0, STOP: 3}}
    states = set()
    states.add('STOP')
    
    prev_state = 'START'
        
    for line in lines:
        split_line = line.split()
        
        if prev_state not in transition_dict:
            transition_dict[prev_state] = {}
                
        # can only be START or STOP
        if len(split_line) < 2:
            if 'STOP' not in transition_dict[prev_state]:
                transition_dict[prev_state]['STOP'] = 0
            
            transition_dict[prev_state]['STOP'] += 1
            prev_state = 'START'
        
        # processing the sentence
        elif len(split_line) == 2:
            curr_state = split_line[1]
            states.add(curr_state)
           
            if curr_state not in transition_dict[prev_state]:
                transition_dict[prev_state][curr_state] = 0
            
            transition_dict[prev_state][curr_state] += 1
            prev_state = curr_state
    
    for k, v in transition_dict.items():
        for state in states:
            if state not in v:
                transition_dict[k][state] = 0
    
    return transition_dict

def get_transition_params(transition_dict, u, v):
    
    if u not in transition_dict:
        raise Exception("State u not in transition dict")
        
    if v not in transition_dict[u]:
        raise Exception("State v not in transition dict")
    
    state_data = transition_dict[u]
    
    count_u_to_v = state_data[v] # count(u,v)
    count_u = sum(state_data.values()) # count(u)
            
    return count_u_to_v / count_u

# Helper Functions
def log(m):
    if isinstance(m, float) or isinstance(m, int):
        return -np.inf if m == 0 else np.log(m)
    
    m = np.clip(m, 1e-32, None)
    x = np.log(m)
    
    x[x <= np.log(1e-32)] = -np.inf
    
    return x

def obtain_all_obs(emission_dict):
    """
    Obtain all distinct observations words in the emission_dict.
    Purpose: This helps us identify words in Test Set that do not exist in the Training Set (or the emission_dict)
    Returns - Set of Strings.
    """
    all_observations = set()
    
    for s_to_obs_dict in emission_dict.values():
        for obs in s_to_obs_dict.keys():
            all_observations.add(obs)
            
    return all_observations

def preprocess_sentence(sentence, training_set_words):
    """
    sentence - a list of Strings (words or observations)
    Returns - a list of Strings, where Strings not in training_set_words are replaced by "#UNK#"
    """
    return [ word if word in training_set_words else "#UNK#" for word in sentence ]

def train(filename):
    """
    Returns - A 2-tuple (Dict, Dict): emission and transition parameters
    """
    return train_emission(filename), train_transition(filename)

def viterbi(emission_dict, transition_dict, sentence, is_preprocessed):
    """
    Dynamic Programming approach (Viterbi algorithm) to generate state sequence given a sentence.
    emission_dict - Dictionary. Emission parameters generated from training data.
    transition_dict - Dictionary. Transition parameters generated from training data.
    sentence - list of Strings (input words or observations)
    is_preprocessed - boolean. True if variable sentence is preprocessed (unknown words (not in train) changed to "#UNK#")
    Returns - List of Strings - (Predicted sequence of states corresponding to sentence).
    """
    
    all_states = list(emission_dict.keys())
    
    proc_sent = sentence
    if not is_preprocessed:
        training_set_words = obtain_all_obs(emission_dict)
        proc_sent = preprocess_sentence(sentence, training_set_words)
    proc_sent = ["start"] + proc_sent + ["stop"]
    
    n = len(proc_sent)

    # Pi np
    P = np.ones( (len(proc_sent), len(all_states))) * -np.inf
    # Backpointer np
    B = [ [ None for x in all_states ] for y in proc_sent ]
    
    # Helper functions for recursive step
    a = lambda u, v: get_transition_params(transition_dict, u, v)
    b = lambda state, obs: get_emission_params_fixed(emission_dict, state, obs, k=0.5)
    
    # Base Case at j=1
    t = np.array([ a('START', v) for v in all_states ])
    e = np.array([ b(v, proc_sent[1]) for v in all_states ])
    P[1, :] = log(t) + log(e)
    B[1] = [ "START" for row in B[1] ]

    # Recursive Forward Step
    for j in range(2, n-1): # Going right the columns (obs)
        x = proc_sent[j]  # Obtain j'th word in the (processed) sentence

        for row_no, v in enumerate(all_states): # Going down the rows (states)
            transitions = np.array([ a(u, v) for u in all_states ])
            prev_scores = P[j-1, :] + log(transitions) 
            top = prev_scores.argmax()
            P[j,row_no] = prev_scores[top] + log(b(v,x))
            B[j][row_no] = all_states[top]
            if P[j,row_no] == -np.inf:
                B[j][row_no] = None

    # Termination: j=n-1. Note that proc_sent[n-1] give us the last word in sentence.
    j = n-1
    transitions = np.array([ a(u, "STOP") for u in all_states ])
    previous_scores = P[j-1] + log(transitions)
    last_state = all_states[previous_scores.argmax()]

    # Backtrack
    state_seq = ['STOP'] + [last_state]
    for j in range(n-2, 0, -1):
        curr_state = state_seq[-1]
        curr_state_row_no = all_states.index(curr_state)    
        prev_state = B[j][curr_state_row_no]

        if prev_state == None: # edge case
            return ['O'] * (n-2)
            break

        state_seq.append(prev_state)

    state_seq = state_seq[::-1][1:-1]  # reverse and drop START, STOP
    return state_seq

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage: python hmm_part3.py EN")
        exit()

    dataset = sys.argv[1]
    if dataset not in ['EN', 'SG', 'CN']:
        print("Second argument must be one of the following: ['EN', 'SG, 'CN']")
        exit()
    
    print(f"Evaluating on {dataset}.")
    
    in_file = f"../{dataset}/dev.in"
    train_file = f"../{dataset}/train"
    out_file = f"../{dataset}/dev.p3.out"
    
   # Train
    emission_dict, transition_dict = train(train_file)

    # Obtain all distinct words in Training Set
    training_set_words = obtain_all_obs(emission_dict)
    
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
            sent_proc = preprocess_sentence(sent, training_set_words)
            # obtain processed sentence's predicted state seq (list of corresponding predicted states for each word in sent)
            sent_state_sequence = viterbi(emission_dict, transition_dict, sent_proc, is_preprocessed=True)

            for word, state in zip(sent, sent_state_sequence):
                outf_h.write(word + ' ' + state)
                outf_h.write("\n") # newline for each word
            outf_h.write("\n") # another newline when end of sentence

            # Reset sentence list
            sent = []
    
    outf_h.close()