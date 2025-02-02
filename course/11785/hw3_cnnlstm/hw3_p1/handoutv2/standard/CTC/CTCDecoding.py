import numpy as np


class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set

    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """
        
        decoded_path = []
        blank = 0
        path_prob = 1

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)
        seq_len = y_probs.shape[1]
        for t in range(seq_len):
            max_id = np.argmax(y_probs[:,t,0])
            max_p = y_probs[max_id,t,0]
            path_prob *=max_p

            if max_id != blank and (len(decoded_path) == 0 or decoded_path[-1] != self.symbol_set[max_id - 1]):
                decoded_path.append(self.symbol_set[max_id-1])
        decoded_path = ''.join(decoded_path)
        return decoded_path, path_prob




class BeamSearchDecoder(object):
    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """
        self.symbol_set = ['']+ symbol_set  # including the blank symbol.
        self.beam_width = beam_width

    def extend_blank(self,t):
        new_path_dict={}
        for path,prob in self.blank_path_probs.items():
            new_path_dict[path] = prob*self.y_probs[0,t]
        for path,prob in self.symbol_path_probs.items():
            if path in new_path_dict.keys():
                new_path_dict[path] += prob * self.y_probs[0,t]   # duplicate merge
            else:
                new_path_dict[path] = prob * self.y_probs[0,t]
        return new_path_dict
    
    def extend_symbol(self,t):
        new_path_dict={}

        for path,prob in self.blank_path_probs.items():
            pre_prob = prob
            for i,s in enumerate(self.symbol_set):
                if i==0:
                    continue
                new_path = path + s
                if new_path in new_path_dict.keys():  # duplicate merge
                    new_path_dict[new_path] += pre_prob *self.y_probs[i,t]
                else:
                    new_path_dict[new_path] = pre_prob*self.y_probs[i,t]
        for path,prob in self.symbol_path_probs.items():
            pre_prob = prob
            for i, s in enumerate(self.symbol_set):
                if i == 0:
                    continue
                if s != path[-1]:
                    new_path = path + s
                else:
                    new_path = path
                if new_path in new_path_dict.keys():
                    new_path_dict[new_path] += pre_prob * self.y_probs[i, t]
                else:
                    new_path_dict[new_path] = pre_prob * self.y_probs[i, t]
        
        return new_path_dict

    def beam_select(self):
        # only select top-k for later step
        sorted_value_list = sorted(list(self.blank_path_probs.values())+list(self.symbol_path_probs.values()),reverse=True)
        if len(sorted_value_list) <= self.beam_width:
            pass
        else:
            new_path_dict={}
            threshold= sorted_value_list[self.beam_width]
            for path,prob in self.blank_path_probs.items():
                if prob >threshold:
                    new_path_dict[path]= prob
            self.blank_path_probs = new_path_dict

            new_path_dict={}
            for path,prob in self.symbol_path_probs.items():
                if prob >threshold:
                    new_path_dict[path]= prob
            self.symbol_path_probs = new_path_dict

    def decode(self, y_probs):
        """
        Perform beam search decoding for CTC.

        Parameters:
        -----------
        y_probs : np.ndarray
            The probability distribution over the symbols at each time step.
            Shape: (num_symbols + 1, seq_length)
            The first index corresponds to the blank symbol.

        Returns:
        --------
        best_path : str
            The decoded sequence with the highest probability.

        final_path_scores : dict
            A dictionary mapping decoded sequences to their probabilities.
        """
        T = y_probs.shape[1]  # Length of the input sequence
        blank = 0
    
        # Initialize paths end with blank at t = 0
        self.y_probs = y_probs[:,:,0]
        self.blank_path_probs = {'': 1}  # P of paths ending with blank
        self.symbol_path_probs = {}  # P of paths ending with a symbol

        #iterate with t
        for t in range(T):
            #beam select:
            self.beam_select()
            # extend path
            self.blank_path_probs,self.symbol_path_probs = self.extend_blank(t),self.extend_symbol(t)

        #merge path ends with blank and symbol
        # print(self.blank_path_probs)
        # print(self.symbol_path_probs)
        Merge_path = self.blank_path_probs
        for path,prob in self.symbol_path_probs.items():
            if path in Merge_path.keys():
                Merge_path[path]+=prob  #merge two path by adding their probs together
            else:
                Merge_path[path] = prob
        # print(Merge_path)
        # get the best path
        sorted_Merge_path = dict(sorted(Merge_path.items(),key= lambda items:items[1],reverse= True))
        best_path = list(sorted_Merge_path.keys())[0]

        return best_path,Merge_path
    
            





        

        