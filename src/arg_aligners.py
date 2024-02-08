import utils
import json
import re
import aligners as Al

class LemmaMatch:
    def execute(self,texts, nlp, config):
        
        #arg id2translation
        args_en_pt_f = open(config["id2translation"])
        args_en_pt_dict = json.load(args_en_pt_f)

        #arg multiple translations
        args_en_pt_pt_micro_translations_lemma_f = open(config["translations_path"])
        args_en_pt_pt_micro_translations_lemma = json.load(args_en_pt_pt_micro_translations_lemma_f)


        total = 0
        normal_match = 0
        lemma_match = 0
        translations_match = 0
        not_found = 0

        for ti, t in enumerate(texts):
            sent_doc = nlp(t["sentence_pt"])
            sent_lemma_tokens = utils.normalizeTokens([token.lemma_.strip().lower()  for token in sent_doc])
            sent_lemma = " ".join(sent_lemma_tokens)
            for ei, e in enumerate(t["golden-event-mentions"]):
                for ai, a in enumerate(e["arguments"]):
                    total +=1
                    #if a["text_pt"] in t["sentence_pt"]:
                    if re.search(r"\b"+re.escape(a["text_pt"])+r'\b',t["sentence_pt"]): ## normal match
                        normal_match += 1
                    else:
                        argument_doc = nlp(a["text_pt"])
                        argument_lemma_tokens = utils.normalizeTokens([token.lemma_.strip().lower() for token in argument_doc])
                        argument_lemma = " ".join(argument_lemma_tokens)
                        argument_tokens = [x.text for x in argument_doc]

                        if re.search(r"\b"+re.escape(argument_lemma)+r'\b',sent_lemma): # lemma match
                            lemma_match+=1
                            arg_span = utils.contains(sent_lemma_tokens,argument_lemma_tokens,nlp)
                            if arg_span:
                                start, end = arg_span
                                texts[ti]["golden-event-mentions"][ei]["arguments"][ai]["previous_pt"] = a["text_pt"]
                                texts[ti]["golden-event-mentions"][ei]["arguments"][ai]["text_pt"] = ''.join([token.text_with_ws for token in sent_doc][start:end+1]).strip()
                    
                                
                        else:   # multiple translation match
                            id = a["entity-id"]
                            en_args = args_en_pt_dict[id]["en"]
                            
                            pt_translations_lemma = args_en_pt_pt_micro_translations_lemma[en_args]
                            arg_span = utils.contains(sent_lemma_tokens,pt_translations_lemma,nlp)
                            if arg_span:
                                translations_match +=1
                                start, end = arg_span
                                texts[ti]["golden-event-mentions"][ei]["arguments"][ai]["previous_pt"] = a["text_pt"]
                                texts[ti]["golden-event-mentions"][ei]["arguments"][ai]["text_pt"] = ''.join([token.text_with_ws for token in sent_doc][start:end+1]).strip()
                    
                            else:
                                not_found += 1
                                texts[ti]["golden-event-mentions"][ei]["arguments"][ai]["failed"] = -1
        return texts
    

class FuzzyMatch:
    def execute(self, texts, nlp):
        for ti, t in enumerate(texts):
            for ei, e in enumerate(t["golden-event-mentions"]):
                for ai, a in enumerate(e["arguments"]):
                    #if "text_0" not in a and a["text"].lower() not in e["scope"].lower():
                    if "failed" in a:
                        arg_3 = Al.align_2(t["sentence_pt"], a["text_pt"], nlp)
                        arg_4 = Al.align_3(t["sentence_pt"], a["text_pt"], nlp)
                        texts[ti]["golden-event-mentions"][ei]["arguments"][ai]["text_3"] = arg_3
                        texts[ti]["golden-event-mentions"][ei]["arguments"][ai]["text_4"] = arg_4
            #print(ti)
        return texts


class WordAlignerMatch:
    def execute(self,texts, nlp):
        count_method = {"gestalt": 0, "levenstein": 0, "word_aligner": 0, "not_found":0}
        for ti, t in enumerate(texts):
            for ei, e in enumerate(t["golden-event-mentions"]):
                for ai, a in enumerate(e["arguments"]):
                    if "failed" in a:
                        en_arg = texts[ti]["golden-event-mentions"][ei]["arguments"][ai]["text"]
                        en_sent = texts[ti]["sentence"]
                        pt_sent = texts[ti]["sentence_pt"]
                        res = Al.wordAligner(en_sent,pt_sent,en_arg, nlp)
                        texts[ti]["golden-event-mentions"][ei]["arguments"][ai]["aligned-bert"] = res

                        arg_3 = texts[ti]["golden-event-mentions"][ei]["arguments"][ai]["text_3"]
                        arg_4 = texts[ti]["golden-event-mentions"][ei]["arguments"][ai]["text_4"]
                        texts[ti]["golden-event-mentions"][ei]["arguments"][ai]["previous_pt"] = texts[ti]["golden-event-mentions"][ei]["arguments"][ai]["text_pt"]
                        aligned_arg, method  = Al.chooseArg([arg_4,arg_3,res],en_arg,nlp)
                        count_method[method] += 1
                        texts[ti]["golden-event-mentions"][ei]["arguments"][ai]["text_pt"]= aligned_arg.strip()
        return texts


