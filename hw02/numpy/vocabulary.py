import os
import math
import numpy as np


class VocabularyWord:
    def __init__(self, word, count = 0):
        self.word = word
        self.count = count

    def __repr__(self):
        return f"VocabularyWord(word='{self.word}', count={self.count})"


class Vocabulary:
    MIN_COUNT_ = 5
    
    def __init__(self):
        self.word2index = {}
        self.train_words_ = 0
        self.min_reduce_ = 1
        
        # Embeddings
        self.embedding_size_ = None
        self.etarget = None
        self.econtext = None


    def read_from_file(self, read_fname):
        """
        Reads vocabulary from a file containing "word count" lines.
        """
        self.words = []
        self.word2index = {}
        
        # It reads lines: word count
        try:
            with open(read_fname, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        word = parts[0]
                        try:
                            count = int(parts[1])
                            self.add_word(word)
                            self.words[-1].count = count
                        except ValueError:
                            pass
                            
            self.sort_vocabulary()
            
            print()
            print(f"Vocabulary size: {len(self.words)}")
            print(f"Words in the training file: {self.train_words_}")
            
        except FileNotFoundError:
            print(f"Error: Vocabulary file {read_fname} not found.")


    def save_to_file(self, save_fname):
        """
        Saves vocabulary to a file.
        Format: word count
        """
        with open(save_fname, 'w', encoding='utf-8') as f:
            for word_obj in self.words:
                f.write(f"{word_obj.word} {word_obj.count}\n")


    def load_embeddings(self, input_fname):
        """
        Loads embeddings from a file.
        Format:
        N D
        word1 val1 val2 ...
        """
        try:
            with open(input_fname, 'r', encoding='utf-8', errors='replace') as f:
                header = f.readline().strip().split()
                if not header:
                    return
                    
                nlines = int(header[0])
                size = int(header[1])
                
                self.embedding_size_ = size

                # We need to ensure etarget is ready.
                if self.etarget is None or self.etarget.shape[0] != len(self.words):
                    self.etarget = np.zeros((len(self.words), self.embedding_size_), dtype=np.float32)
                
                for _ in range(nlines):
                    line = f.readline()
                    if not line:
                        break
                    
                    parts = line.strip().split()
                    if not parts:
                        continue
                        
                    word = parts[0]
                    vec_vals = parts[1:]
                    
                    idx = self.get_word_index(word)
                    if idx != -1:
                        # Parse vector
                        if len(vec_vals) == self.embedding_size_:
                            vec = np.array(vec_vals, dtype=np.float32)
                            self.etarget[idx] = vec
                        else:
                            # Handle mismatch?
                            pass
                            
        except FileNotFoundError:
             print(f"Error: Embedding file {input_fname} not found.")



    def learn_from_file(self, train_fname):
        self.words = []
        self.word2index = {}
        
        # Add </s>
        self.add_word("</s>")
        
        self.train_words_ = 0
        
        print(f"Learning vocabulary from training file {train_fname}")
        
        with open(train_fname, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                tokens = line.strip().split()
                if not tokens:
                    continue
                
                for token in tokens:
                    self.train_words_ += 1
                    pos = self.word2index.get(token, -1)
                    if pos == -1:
                        pos = self.add_word(token)
                    self.words[pos].count += 1
                    
                    if len(self.words) > 21000000: # Rough limit (hash_capacity * 0.7)
                         self.reduce_vocabulary()
                
                # End of line </s>
                self.words[0].count += 1
                self.train_words_ += 1
                
                if self.train_words_ % 100000 == 0:
                     print(f"{self.train_words_ // 1000}K ", end='', flush=True)

        print()
        print(f"Vocabulary size: {len(self.words)}")
        print(f"Words in the training file: {self.train_words_}")
        
        self.sort_vocabulary()
        
        print()
        print(f"Vocabulary size: {len(self.words)}")
        print(f"Words in the training file: {self.train_words_}")


    def add_word(self, word):
        vword = VocabularyWord(word)
        self.words.append(vword)
        pos = len(self.words) - 1
        self.word2index[word] = pos
        return pos

    def reduce_vocabulary(self):
        # Remove words with count <= min_reduce_
        # Keep </s> at index 0
        new_words = []
        new_word2index = {}
        
        # Always keep </s>
        new_words.append(self.words[0])
        new_word2index[self.words[0].word] = 0
        
        for i in range(1, len(self.words)):
            if self.words[i].count > self.min_reduce_:
                new_pos = len(new_words)
                new_words.append(self.words[i])
                new_word2index[self.words[i].word] = new_pos
                
        self.words = new_words
        self.word2index = new_word2index
        self.min_reduce_ += 1

    def sort_vocabulary(self):
        # Sort by count desc, but keep </s> at 0
        # Slice [1:] sort
        # Primary sorting logic based on count desc
        
        # Python sort is stable.
        part_to_sort = self.words[1:]
        # Sort descending by count
        part_to_sort.sort(key=lambda w: w.count, reverse=True)
        
        self.words = [self.words[0]] + part_to_sort
        
        # Rebuild index and prune MIN_COUNT_
        new_words = []
        new_word2index = {}
        
        # </s> always in
        new_words.append(self.words[0])
        new_word2index[self.words[0].word] = 0
        self.train_words_ = self.words[0].count
        
        for i in range(1, len(self.words)):
            if self.words[i].count >= self.MIN_COUNT_:
                new_pos = len(new_words)
                new_words.append(self.words[i])
                new_word2index[self.words[i].word] = new_pos
                self.train_words_ += self.words[i].count
            else:
                # Since sorted, can break early?
                # Since sorted desc, once < MIN_COUNT, rest are also <.
                # So we can break.
                pass
        
        self.words = new_words
        self.word2index = new_word2index


    def init_embeddings(self):
        # Init etarget
        # Random range [-0.5/size, 0.5/size]
        # (next_random & 0xFFFF) / (float)65536 - 0.5) / size
        
        vocab_size = len(self.words)
        np.random.seed(1)
        
        low = -0.5 / self.embedding_size_
        high = 0.5 / self.embedding_size_
        
        self.etarget = np.random.uniform(low, high, (vocab_size, self.embedding_size_)).astype(np.float32)
        
        # Init econtext to zeros
        self.econtext = np.zeros((vocab_size, self.embedding_size_), dtype=np.float32)


    def save_embeddings(self, output_fname, use_etarget=True):
        assert self.etarget is not None
        
        with open(output_fname, 'w') as f:
            f.write(f"{len(self.words)} {self.embedding_size_}\n")
            for i, word_obj in enumerate(self.words):
                f.write(word_obj.word + " ")
                vec = self.etarget[i] if use_etarget else self.econtext[i]
                # Join with spaces
                f.write(" ".join(map(str, vec)))
                f.write("\n")


    def get_word_index(self, word):
        return self.word2index.get(word, -1)


    def similarity(self, word1, word2):
        """
        Computes cosine similarity between two words, using their `etarget` embeddings by default.
        Returns 0 if word not found.
        """
        idx1 = self.get_word_index(word1)
        idx2 = self.get_word_index(word2)
        if idx1 == -1 or idx2 == -1:
            return 0

        v1 = self.etarget[idx1]
        v2 = self.etarget[idx2]

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0

        return float(np.dot(v1, v2) / (n1 * n2))


    def get_similar_words(self, word, k=15):
        """
        Finds and returns list of top K similar words; return [] if word not found.
        """

        idx = self.get_word_index(word)
        if idx == -1:
            return []

        vocab_size = len(self.words)
        if vocab_size == 0:
            return []

        k = int(k)
        if k <= 0:
            return []

        # Normalize embeddings (cosine similarity)
        E = self.etarget
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        En = E / norms

        q = En[idx]
        scores = En @ q

        # Exclude the query word and sentence token if present
        scores[idx] = -np.inf
        if vocab_size > 0:
            scores[0] = -np.inf

        kk = min(k, vocab_size - 1) if vocab_size > 1 else 0
        if kk <= 0:
            return []

        top_idx = np.argpartition(-scores, kk)[:kk]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        return [(float(scores[i]), int(i)) for i in top_idx]


    def get_analogy_words(self, words, k=15):
        """
        Finds and returns list of top K words for the analogy triplet in `words`; return [] if a word is not found.
        """

        if len(words) < 3:
            return []

        w1, w2, w3 = words[0], words[1], words[2]
        idx1 = self.get_word_index(w1)
        idx2 = self.get_word_index(w2)
        idx3 = self.get_word_index(w3)

        if idx1 == -1 or idx2 == -1 or idx3 == -1:
            return []

        vocab_size = len(self.words)
        if vocab_size == 0:
            return []

        k = int(k)
        if k <= 0:
            return []

        # Normalize embeddings (cosine similarity)
        E = self.etarget
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        En = E / norms

        v1 = En[idx1]
        v2 = En[idx2]
        v3 = En[idx3]

        # Analogy vector: v2 - v1 + v3
        q = v2 - v1 + v3
        qn = np.linalg.norm(q)
        if qn == 0:
            return []
        q = q / qn

        scores = En @ q

        # Exclude input words and sentence token if present
        scores[idx1] = -np.inf
        scores[idx2] = -np.inf
        scores[idx3] = -np.inf
        scores[0] = -np.inf

        kk = min(k, vocab_size - 1) if vocab_size > 1 else 0
        if kk <= 0:
            return []

        top_idx = np.argpartition(-scores, kk)[:kk]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        return [(float(scores[i]), int(i)) for i in top_idx]
