
import sys
import argparse
from lm_embed_model import LMEmbedModel

def main():
    parser = argparse.ArgumentParser(description="Word2Vec NumPy Implementation")
    
    parser.add_argument("-train", "--train", dest="train", help="Use text data from <file> to train the model", required=True)
    parser.add_argument("-output", "--output", dest="output", help="Use <file> to save the resulting word vectors", required=True)
    parser.add_argument("-size", "--size", dest="size", type=int, default=100, help="Set size of word vectors; default is 100")
    parser.add_argument("-context", "--context", dest="context", type=int, default=5, help="Set max skip length between words; default is 5")
    parser.add_argument("-subsample", "--subsample", dest="subsample", type=float, default=1e-3, help="Set threshold for occurrence of words")
    parser.add_argument("-negative", "--negative", dest="negative", type=int, default=5, help="Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)")
    parser.add_argument("-iter", "--iter", dest="iter", type=int, default=5, help="Run more training iterations (default 5)")
    parser.add_argument("-alpha", "--alpha", dest="alpha", type=float, default=0.025, help="Set the starting learning rate; default is 0.025 for skip-gram")
    parser.add_argument("-mincount", "--mincount", dest="mincount", type=int, default=5, help="This will discard words that appear less than <int> times; default is 5")
    
    
    # Optional args for saving/reading vocabulary
    parser.add_argument("-savevocab", "--savevocab", dest="savevocab", help="The vocabulary will be saved to <file>")
    parser.add_argument("-readvocab", "--readvocab", dest="readvocab", help="The vocabulary will be read from <file>")

    args = parser.parse_args()

    # Set parameters
    # Set parameters using static members for configuration in Vocabulary
    
    # We need to import Vocabulary to set static configs if we kept that design
    from vocabulary import Vocabulary
    Vocabulary.MIN_COUNT_ = args.mincount
    Vocabulary.CONTEXT_SIZE_ = args.context
    
    model = LMEmbedModel(
        negatives = args.negative,
        iter_num = args.iter,
        subsample_rate = args.subsample,
        alpha = args.alpha
    )
    
    if args.readvocab:
        model.read_vocabulary(args.readvocab)
    else:
        model.learn_vocabulary(args.train)
        
    if args.savevocab:
        model.save_vocabulary(args.savevocab)

    model.vocabulary_.embedding_size_ = args.size
    
    model.train(args.train)
    
    model.save_embeddings(args.output + '.1', use_etarget = True)
    model.save_embeddings(args.output + '.2', use_etarget = False)

if __name__ == "__main__":
    main()
