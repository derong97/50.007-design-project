from tqdm.auto import tqdm
import numpy as np
import sys

# Emission
def train_emission(filename):
    """
    Returns - a dictionary containing emission parameters
    """# 
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

# Train
def train(filename):
    """
    Returns - A 2-tuple (Dict, Dict): emission and transition parameters
    """
    return train_emission(filename), train_transition(filename)

def topk_viterbi(emission_dict, transition_dict, sentence, is_preprocessed, k=3):
    all_states = list(emission_dict.keys())
    
    proc_sent = sentence
    if not is_preprocessed:
        training_set_words = obtain_all_obs(emission_dict)
        proc_sent = preprocess_sentence(sentence, training_set_words)
    proc_sent = ["start"] + proc_sent + ["stop"]
    
    n = len(proc_sent)
    
    # Pi Table
    P = np.ones( (len(proc_sent), len(all_states), k) ) * -np.inf
    
    # Backpointer Table
    B = [ [ [None] * k for x in all_states ] for y in proc_sent ]
    
    a = lambda u, v: get_transition_params(transition_dict, u, v)
    b = lambda state, obs: get_emission_params_fixed(emission_dict, state, obs, k=0.5)

    # Base Case at j=1
    t = np.array([ a('START', v) for v in all_states ])
    e = np.array([ b(v, proc_sent[1]) for v in all_states ])
    P[1, :, 0] = log(t) + log(e)

    B[1] = [ ["START"] + [None] * (k-1) for row in B[1] ]
    
    # Recursive Forward Step for j=2 to n-2 (inclusive). Where n = len(proc_sent) => Layer n is the last layer (STOP).
    # At layer n-1, we perform termination step, as layer n is STOP. 
    # Recall that proc_sent includes 'start' and 'stop'

    for j in range(2, n-1): # Going right the columns (obs)
        x = proc_sent[j]  # Obtain j'th word in the (processed) sentence

        for row_no, v in enumerate(all_states): # Going down the rows (states)
            transitions = np.array([ a(u, v) for u in all_states ])
            previous_all_scores = (P[j-1, :] + log(transitions[:, None])).flatten()
            topk = previous_all_scores.argsort()[::-1][:k]
            P[j,row_no] = previous_all_scores[topk] + log(b(v,x))
            B[j][row_no] = [ all_states[pos // k] for pos in topk ]
            
            for i, sub_k in enumerate(P[j,row_no]):
                if sub_k == -np.inf:
                    B[j][row_no][i] = None
            
    # Termination: j=n-1. Note that proc_sent[n-1] give us the last word in sentence.
    j = n-1
    transitions = np.array([ a(u, "STOP") for u in all_states ])
    previous_all_scores = (P[j-1] + log(transitions[:, None])).flatten()
    final_topk = previous_all_scores.argsort()[::-1][:k]
    final_scores = previous_all_scores[final_topk]
    
    # top k parent STATES preceding the STOP state. By top k, means top k best scores.
    final_topk_pos = [ all_states[pos // k] for pos in final_topk ]
    
    # Backtrack
    state_seq = ['STOP']
    
    prev_states = final_topk_pos
    prev_state = prev_states[-1] # you already know you want the kth best, which is the last
    prev_row_no = all_states.index(prev_state)
    curr_score = final_scores[-1]
    
    # from n-2 to n-1 (STOP)
    j = n-1
    for i in range(k):
        prev_score = P[j-1, prev_row_no][i]
        if prev_score + log(a(prev_state, "STOP")) == curr_score:
            curr_score = prev_score
            curr_idx = i
            state_seq.append(prev_state)
            break
    
    for j in range(n-2, 1, -1):
        x = proc_sent[j]
        curr_state = state_seq[-1]
        curr_row_no = all_states.index(curr_state)
        prev_state = B[j][curr_row_no][curr_idx]
        
        if prev_state == None: # No possible transition to STOP. Edge case.
            state_seq = ['O'] * (n-2)
            return P, B, state_seq
        
        prev_row_no = all_states.index(prev_state)
        
        for i in range(k):
            prev_score = P[j-1, prev_row_no][i]
            if prev_score + log(a(prev_state, curr_state)) + log(b(curr_state, x)) == curr_score:
                curr_score = prev_score
                curr_idx = i
                state_seq.append(prev_state)
                break
    
    state_seq.append("START") # START will throw an error because it has no index in all_states
    state_seq = state_seq[::-1][1:-1]  # reverse and drop START, STOP
    
    return P, B, state_seq

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage: python hmm_part4.py EN")
        exit()

    dataset = sys.argv[1]
    if dataset not in ['EN', 'SG', 'CN']:
        print("Second argument must be one of the following: ['EN', 'SG, 'CN]")
        exit()
    
    print(f"Evaluating on {dataset}.")
    
    in_file = f"../dataset/{dataset}/dev.in"
    train_file = f"../dataset/{dataset}/train"
    out_file = f"../dataset/{dataset}/dev.p4.out"
    
   # Train
    emission_dict, transition_dict = train(train_file)

    # Obtain all distinct words in Training Set
    training_set_words = obtain_all_obs(emission_dict)
    
    # Create file handler to write to /dev.p4.out
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
            _, _, sent_state_sequence = topk_viterbi(emission_dict, transition_dict, sent_proc, is_preprocessed=True, k=3)

            for word, state in zip(sent, sent_state_sequence):
                outf_h.write(word + ' ' + state)
                outf_h.write("\n") # newline for each word
            outf_h.write("\n") # another newline when end of sentence

            # Reset sentence list
            sent = []
    
    outf_h.close()