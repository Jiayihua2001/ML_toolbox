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

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """

        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
                        batch size for part 1 will remain 1, but if you plan to use your
                        implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        T = y_probs.shape[1]
        beam = [(tuple(), 1.0)]  # Initialize beam 

        for t in range(T):
            new_beam = []
            for path, path_prob in beam:
                for i in range(y_probs.shape[0]):
                    new_path = path + (i,)
                    new_path_prob = path_prob * y_probs[i, t, 0]
                    new_beam.append((new_path, new_path_prob))

            # sort and select top-k paths according to beam_width
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:self.beam_width]

        # find best path and score
        best_path, best_path_score = max(beam, key=lambda x: x[1])

        # convert to symbol and to str
        forward_path = []
        blank = 0
        for i in range(len(best_path)):
            # ignoring blanks and repeated symbols
            if best_path[i] != blank and (i == 0 or best_path[i] != best_path[i - 1]):
                forward_path.append(self.symbol_set[best_path[i] - 1])

        forward_path = ''.join(forward_path)
    
        merged_path_scores = {''.join(self.symbol_set[i - 1] for i in path if i != blank): prob for path, prob in beam}

        return forward_path, merged_path_scores
