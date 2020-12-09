from tqdm.auto import tqdm
import sys

# Estimate emission parameters from the training set using MLE
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

def get_emission_params(emission_dict, state, obs):
    
    if state not in emission_dict:
        raise Exception("State not in emission dict")
    
    state_data = emission_dict[state]
    
    if obs not in state_data:
        raise Exception("Word did not appear in training data")
    
    count_y_to_x = state_data[obs] # count(y -> x)
    count_y = sum(state_data.values()) # count(y)
    
    return count_y_to_x / count_y

# Modify the computation of emission probabilities
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

# Implement a simple system that produces the tag for each word `x` in the sequence
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
    sentence - a list of Strings (word or observations)
    Returns - a list of Strings, where Strings not in training_set_words are replaced by "#UNK#"
    """
    return [ word if word in training_set_words else "#UNK#" for word in sentence ]

def label_sequence(sentence, emission_dict):
    """
    sentence - a list of Strings (words or observations).
    emission_dict - a dictionary containing emission parameters
    Returns - list of Strings (corresponding highest prob state for each word)
    """
    
    all_states = list(emission_dict.keys()) # all distinct states
    
    sequence = [] # aka tags
    
    for word in sentence:
        emission_state = { state: get_emission_params_fixed(emission_dict, state, word) for state in all_states }
        sequence.append(max(emission_state, key=lambda state: emission_state[state]))
        
    return sequence

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage: python hmm_part2.py EN")
        exit()

    dataset = sys.argv[1]
    if dataset not in ['EN', 'SG', 'CN']:
        print("Second argument must be one of the following: ['EN', 'SG, 'CN]")
        exit()
    
    print(f"Evaluating on {dataset}.")
    
    in_file = f"../dataset/{dataset}/dev.in"
    train_file = f"../dataset/{dataset}/train"
    out_file = f"../dataset/{dataset}/dev.p2.out"
    
    # Train
    emission_dict = train_emission(train_file)

    # Obtain all distinct words in Training Set
    training_set_words = obtain_all_obs(emission_dict)
    
    # Create file handler to write to /dev.p2.out
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
            sent_state_sequence = label_sequence(sent_proc, emission_dict)

            for word, state in zip(sent, sent_state_sequence):
                outf_h.write(word + ' ' + state)
                outf_h.write("\n") # newline for each word
            outf_h.write("\n") # another newline when end of sentence

            # Reset sentence list
            sent = []
    
    outf_h.close()