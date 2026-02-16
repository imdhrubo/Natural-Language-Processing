
import sys
import numpy as np
from vocabulary import Vocabulary

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <in:vocab file> <in:embed file>!")
        sys.exit(1)

    vocab_file = sys.argv[1]
    embed_file = sys.argv[2]
    
    vocab = Vocabulary()
    vocab.read_from_file(vocab_file)
    vocab.load_embeddings(embed_file)
        
    try:
        line = sys.stdin.readline()
        while line:
            stripped_line = line.strip()
            if not stripped_line:
                # Empty line terminates loop.
                break
                
            words = stripped_line.split()
            count = len(words)
            
            if count == 1:
                word = words[0]
                idx = vocab.get_word_index(word)
                if idx == -1:
                    print(f"{word} not in vocabulary!")
                else:
                    top_words = vocab.get_similar_words(word, 15)
                    for score, idx in top_words:
                        print(f"  {vocab.words[idx].word} {score}")
                        
            elif count == 2:
                word1 = words[0]
                word2 = words[1]
                idx1 = vocab.get_word_index(word1)
                
                if idx1 == -1:
                    print(f"{word1} not in vocabulary!")
                    # Go to next iteration
                else:
                    idx2 = vocab.get_word_index(word2)
                    if idx2 == -1:
                        print(f"{word2} not in vocabulary!")
                    else:
                        sim = vocab.similarity(word1, word2)
                        print(sim)
                        
            elif count >= 3:
                word1 = words[0]
                word2 = words[1]
                word3 = words[2]
                
                idx1 = vocab.get_word_index(word1)
                if idx1 == -1:
                    print(f"{word1} not in vocabulary!")
                else:
                    idx2 = vocab.get_word_index(word2)
                    if idx2 == -1:
                        print(f"{word2} not in vocabulary!")
                    else:
                        idx3 = vocab.get_word_index(word3)
                        if idx3 == -1:
                            print(f"{word3} not in vocabulary!")
                        else:
                            top_words = vocab.get_analogy_words([word1, word2, word3], 15)
                            for score, idx in top_words:
                                print(f"  {vocab.words[idx].word} {score}")
            
           
            line = sys.stdin.readline()
            
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
