class CharacterNGramBiDirectionalModel:
    def __init__(self, n, dictionary_list):
        self.n = n
        self.words = dictionary_list
        self.universal_ch_set = [chr(ord('a') + i) for i in range(26)]
        self.prev_context_candidates = {}
        self.post_context_candidates = {}
        self.prev_context_candidates_counts = {}
        self.post_context_candidates_counts = {}
        self.__build()
        
    def __build(self):
        for word in self.words:
            # Prev Context Calc
            tokens = ['<start>']*(self.n-1) + list(word)
            for i in range(self.n-1, len(tokens)):
                prev_context = tuple([tokens[i-j-1] for j in range(self.n-2, -1, -1)])
                next_ch = tokens[i]
                ngram = (prev_context, next_ch)
                current_count = self.prev_context_candidates_counts.get(ngram, 0)
                self.prev_context_candidates_counts[ngram] = current_count + 1
                current_candidates = self.prev_context_candidates.get(prev_context, [])
                current_candidates.append(next_ch)
                self.prev_context_candidates[prev_context] = current_candidates
            # Post Context Calc
            tokens = list(word) + ['<end>']*(self.n-1)
            for i in range(len(tokens)-self.n, -1, -1):
                post_context = tuple([tokens[i+j+1] for j in range(self.n-1)])
                prev_ch = tokens[i]
                ngram = (post_context, prev_ch)
                current_count = self.post_context_candidates_counts.get(ngram, 0)
                self.post_context_candidates_counts[ngram] = current_count + 1
                current_candidates = self.post_context_candidates.get(post_context, [])
                current_candidates.append(prev_ch)
                self.post_context_candidates[post_context] = current_candidates
                
    def calculate_probability(self, prev_context, post_context, ch_set=None):
        if ch_set == None:
            ch_set = self.universal_ch_set
        len_prev_context = len(prev_context)
        tokens = ['<start>']*(self.n-1) + list(prev_context)
        prev_context = tuple(tokens[-self.n+1:])
        len_post_context = len(post_context)
        tokens = list(post_context) + ['<end>']*(self.n-1)
        post_context = tuple(tokens[:len(tokens)-self.n+1])
        probability_dict = {}
        for ch in ch_set:
            # Prev Context Calc
            if "." in prev_context:
                prev_prob = 0
            else:
                ngram = (prev_context, ch)
                context_count = float(len(self.prev_context_candidates.get(prev_context, [])))
                ngram_count = float(self.prev_context_candidates_counts.get(ngram, 0))
                if context_count == 0 or ngram_count == 0:
                    prev_prob = 0
                else:
                    prev_prob = ngram_count/context_count
            # Post Context Calc
            if "." in post_context:
                post_prob = 0
            else:
                ngram = (post_context, ch)
                context_count = float(len(self.post_context_candidates.get(post_context, [])))
                ngram_count = float(self.post_context_candidates_counts.get(ngram, 0))
                if context_count == 0 or ngram_count == 0:
                    post_prob = 0
                else:
                    post_prob = ngram_count/context_count
            if prev_prob == 0:
                probability_dict[ch] = post_prob
            elif post_prob == 0:
                probability_dict[ch] = prev_prob
            else:
                probability_dict[ch] = max([prev_prob,post_prob])
        return probability_dict