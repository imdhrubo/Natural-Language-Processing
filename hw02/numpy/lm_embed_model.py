import math
import numpy as np
from scipy.special import expit
from vocabulary import Vocabulary


class LMEmbedModel:
    UNI_SAMPLE_SIZE_ = 100_000_000 # 1e8
    MAX_SENT_LENGTH_ = 1000

    def __init__(self, negatives, iter_num, subsample_rate, alpha):
        self.vocabulary_ = None
        self.negatives_ = negatives if negatives >= 0 else 5
        self.iter_ = iter_num if iter_num >= 0 else 5
        self.subsample_rate_ = subsample_rate if subsample_rate >= 0 else 1e-3
        self.starting_alpha_ = alpha if alpha >= 0 else 0.025
        self.alpha_ = self.starting_alpha_
        
        self.total_word_count_ = 0
        self.uni_sample_ = None
        
        # Consistent random state for reproduction helper
        self.next_random_ = 1


    @staticmethod
    def get_line(file_path):
        """
        Generator that yields lines of words from a file.
        Yields whitespace-separated tokens.
        """
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                tokens = line.strip().split()
                if not tokens:
                    continue
                
                # Yield tokens in chunks if lines are very long, or simpler: just yield the list of tokens
                # Reads until newline or MAX_SENT_LENGTH_. 
                # Here we can yield sentence by sentence.
                # Accumulates words into a 'line' vector until newline 
                # OR size limit. 
                
                # Ideally we want to yield a list of tokens.
                
                curr_line = []
                for token in tokens:
                    curr_line.append(token)
                    if len(curr_line) >= LMEmbedModel.MAX_SENT_LENGTH_:
                        yield curr_line
                        curr_line = []
                
                if curr_line:
                    yield curr_line 


    def learn_vocabulary(self, train_fname):
        self.vocabulary_ = Vocabulary()
        self.vocabulary_.learn_from_file(train_fname)
        
    def read_vocabulary(self, vocab_fname):
        self.vocabulary_ = Vocabulary()
        self.vocabulary_.read_from_file(vocab_fname)
        
    def save_vocabulary(self, vocab_fname):
        self.vocabulary_.save_to_file(vocab_fname)


    def init_unigram_table(self):
        print("Initializing unigram table...")
        vocab_size = len(self.vocabulary_.words)
        power = 0.75
        
        # Calculate Z
        Z = sum(pow(w.count, power) for w in self.vocabulary_.words)
        
        self.uni_sample_ = np.zeros(self.UNI_SAMPLE_SIZE_, dtype=np.int32)
        
        i = 0
        current_pow = pow(self.vocabulary_.words[i].count, power)
        cdist = current_pow / Z
        
        for a in range(self.UNI_SAMPLE_SIZE_):
            self.uni_sample_[a] = i
            if a / self.UNI_SAMPLE_SIZE_ > cdist:
                i += 1
                if i >= vocab_size:
                    i = vocab_size - 1
                current_pow = pow(self.vocabulary_.words[i].count, power)
                cdist += current_pow / Z


    def train(self, train_fname):
        if self.negatives_ > 0:
            self.init_unigram_table()
            
        self.vocabulary_.init_embeddings()
        
        self.alpha_ = self.starting_alpha_
        self.total_word_count_ = 0
        
        word_count = 0
        last_word_count = 0
        
        csize = Vocabulary.CONTEXT_SIZE_ if hasattr(Vocabulary, 'CONTEXT_SIZE_') else 5 
        
        print(f"Training on {self.vocabulary_.train_words_} words")
        
        train_words = self.vocabulary_.train_words_
        
        for iter_idx in range(self.iter_):
            print(f"Iteration {iter_idx} started.")
            
            # Use generator
            line_gen = self.get_line(train_fname)
            
            for line_tokens in line_gen:
                sampled_words = self.sample_words(line_tokens)

                word_count += len(sampled_words)
                
                for sp, target_idx in enumerate(sampled_words):
                    # target_idx is index in vocabulary
                    
                    self.update_random()
                    b = self.next_random_ % csize
                    
                    # Context window
                    start = max(0, sp - (csize - b))
                    end = min(len(sampled_words) - 1, sp + (csize - b))
                    
                    target_embed = self.vocabulary_.etarget[target_idx]
                    
                    # Accumulate gradient updates for target word.
                    target_update = np.zeros(self.vocabulary_.embedding_size_, dtype=np.float32)
                    
                    for c in range(start, end + 1):
                        if c == sp:
                            continue
                            
                        pos_idx = sampled_words[c]
                        
                        # Positive context word and its labels.
                        context_idxs = [pos_idx]
                        labels = [1]

                        # Negative context words and their labels.   
                        for _ in range(self.negatives_):
                            neg_idx = self.sample_random_word()
                            if neg_idx == pos_idx:
                                continue
                            context_idxs.append(neg_idx)
                            labels.append(0)

                        # YOUR CODE HERE (VECTORIZE IT)
                        # Compute gradients w.r.t. center (target) word, positive and negative context words, use `self.alpha_` as learning rate.
                        # Update `econtext` embeddings for context words,
                        # Accumulate gradient updates for center (target) word in `target_update` 

                        context_idxs_arr = np.asarray(context_idxs, dtype=np.int32)
                        labels_arr = np.asarray(labels, dtype=np.float32)

                        # Gather context embeddings (K x D)
                        context_embeds = self.vocabulary_.econtext[context_idxs_arr]

                        # Scores: (K,)
                        scores = context_embeds @ target_embed

                        # Sigmoid probabilities
                        probs = expit(scores)

                        # Gradient scale: alpha * (label - prob)
                        g = self.alpha_ * (labels_arr - probs)  # (K,)

                        # Update context embeddings:
                        # econtext[c] += g_k * target_embed
                        # Use np.add.at to correctly handle duplicate indices.
                        np.add.at(
                            self.vocabulary_.econtext,
                            context_idxs_arr,
                            g[:, None] * target_embed[None, :]
                        )

                        # Accumulate update for target embedding:
                        # target_update += sum_k g_k * context_embed_k
                        target_update += (g[:, None] * context_embeds).sum(axis=0)

                    # Update `etarget` embedding for target (center) word.
                    self.vocabulary_.etarget[target_idx] += target_update

                if word_count - last_word_count > 10000:
                    self.total_word_count_ += word_count - last_word_count
                    last_word_count = word_count
                    
                    # Update alpha
                    self.alpha_ = self.starting_alpha_ * (1 - self.total_word_count_ / (self.iter_ * train_words + 1))
                    if self.alpha_ < self.starting_alpha_ * 0.0001:
                        self.alpha_ = self.starting_alpha_ * 0.0001
                        
                    print(f"Progress: {self.total_word_count_ // 1000}K words, Alpha: {self.alpha_:.5f}")

            self.total_word_count_ += word_count - last_word_count
            word_count = 0
            last_word_count = 0
            
            # Save periodic
            # self.vocabulary_.save_embeddings(f"embeddings-{iter_idx}.txt.1", True)


    def sample_words(self, line_tokens):
        # Returns list of indices
        words = []
        wtotal = self.vocabulary_.train_words_
        
        for token in line_tokens:
            idx = self.vocabulary_.get_word_index(token)
            if idx == -1:
                continue
                
            vword = self.vocabulary_.words[idx]
            
            # Subsampling
            # formula: rate = (sqrt(count / (sample * total)) + 1) * (sample * total) / count
            
            ran = (math.sqrt(vword.count / (self.subsample_rate_ * wtotal)) + 1) * (self.subsample_rate_ * wtotal) / vword.count
            
            self.update_random()
            
            if ran > (self.next_random_ & 0xFFFF) / 65536.0:
                 words.append(idx)
                 
        return words


    def sample_random_word(self):
        self.update_random()
        idx = (self.next_random_ >> 16) % self.UNI_SAMPLE_SIZE_
        target = self.uni_sample_[idx]
        
        if target == 0:
            target = (self.next_random_ % (len(self.vocabulary_.words) - 1)) + 1
            
        return target
        
    def update_random(self):
        self.next_random_ = self.next_random_ * 25214903917 + 11
        self.next_random_ = self.next_random_ & 0xFFFFFFFFFFFFFFFF # unsigned long long simulation


    def save_embeddings(self, output_fname):
        self.vocabulary_.save_embeddings(output_fname)
