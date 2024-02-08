import yaml
import arg_aligners as Arg_AL
import trg_aligners as Trg_AL
import json
import spacy
import utils
import sys




def align_Triggers(config, texts, nlp):
    pipeline = config['trigger_alignment']['pipeline']

    for method in pipeline:
        if method == 'lemma':
            aligner = Trg_AL.LemmaMatch()
            texts = aligner.execute(texts,nlp,config["trigger_alignment"])
        elif method == 'MTrans': # Mtrans is combined with lemma method since we also calculate the lemma of the translations
            continue
        elif method == 'synonyms':
            continue
        elif method == 'word_aligner':
            aligner = Trg_AL.WordAlignerMatch()
            texts = aligner.execute(texts,nlp)
        else:
            print(f"Invalid trigger alignment method: {method}")
    return texts



def align_Arguments(config, texts, nlp):
    pipeline = config['argument_alignment']['pipeline']

    for method in pipeline:
        if method == 'lemma': 
            aligner = Arg_AL.LemmaMatch()
            texts = aligner.execute(texts, nlp, config["argument_alignment"])
        elif method == 'MTrans': # Mtrans is combined with lemma method since we also calculate the lemma of the translations
            continue
        elif method == 'fuzzy':
            aligner = Arg_AL.FuzzyMatch()
            texts = aligner.execute(texts,nlp)
        elif method == 'word_aligner':
            aligner = Arg_AL.WordAlignerMatch()
            texts = aligner.execute(texts,nlp)                
        else:
            print(f"Invalid argument alignment method: {method}")
    return texts


def loadModel(config):
    nlp = spacy.load(config["spacy_model"])
    prefixes = list(nlp.Defaults.prefixes) +["ex"]+["-"]
    prefix_regex = spacy.util.compile_prefix_regex(prefixes)
    nlp.tokenizer.prefix_search = prefix_regex.search
    return nlp

def dataLoad(config):
    if not sys.argv[1]:
        filename = config['input_path']
    else:
        filename = sys.argv[1]
        
    file_in_aligned = open(filename)
    texts = json.load(file_in_aligned)
    return texts


def dataSave(config,texts):
    if not sys.argv[2]:
        filename = config['output_path']
    else:
        filename = sys.argv[2]
    with  open(filename,"w") as out_f:
        json.dump(texts,out_f,indent=4,ensure_ascii=False)
    

def execute_pipeline(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)["config"]
        texts = dataLoad(config)
        nlp = loadModel(config)
        texts = align_Triggers(config,texts,nlp)
        texts = align_Arguments(config,texts,nlp)
        texts = utils.clean(texts)
        dataSave(config,texts)
        

def main():
    config_file = '../config.yaml'
    execute_pipeline(config_file)

if __name__ == '__main__':
    main()