import re
import json
import utils
import aligners as Al



class LemmaMatch:
    def execute(self,texts, nlp, config):
        #synonyms
        synonyms_br_lemma_f = []
        synonyms_br_lemma = []

        #tigger id2transltion
        trigger_en_pt_f = open(config["id2translation"])
        trigger_en_pt_dict = json.load(trigger_en_pt_f)

        #multiple translation (Microsoft API)
        translations_en_pt_f = open(config["translations_path"])
        trigger_en_pt_translations_lemma = json.load(translations_en_pt_f)


        total = 0
        normal_match = 0
        lemma_match = 0
        translations_match = 0
        synonym_match = 0
        not_found = 0

        for ti, t in enumerate(texts):
            for ei, e in enumerate(t["golden-event-mentions"]):
                total+=1
                if re.search(r"\b"+re.escape(e["trigger"]["text_pt"])+r'\b',t["sentence_pt"]):
                    normal_match +=1
                else:
                    sent_doc = nlp(t["sentence_pt"])
                    trigger_doc = nlp(e["trigger"]["text_pt"])
                    sent_lemma_tokens = utils.normalizeTokens([x.lemma_.strip().lower() for x in sent_doc])
                    trigger_lemma_tokens_aux =[x.lemma_ for x in trigger_doc]
                    trigger_lemma_tokens = utils.normalizeTokens(x.strip().lower() for x in trigger_lemma_tokens_aux)
                    trigger_lemma = " ".join(trigger_lemma_tokens)
                    sent_lemma = " ".join(sent_lemma_tokens)
                    
                    if re.search(r"\b"+re.escape(trigger_lemma)+r'\b',sent_lemma):
                        trigger_span = utils.contains(sent_lemma_tokens,trigger_lemma_tokens,nlp)
                        if trigger_span:
                            start, end = trigger_span
                            texts[ti]["golden-event-mentions"][ei]["trigger"]["previous_pt"] = texts[ti]["golden-event-mentions"][ei]["trigger"]["text_pt"]
                            texts[ti]["golden-event-mentions"][ei]["trigger"]["text_pt"] = ''.join([token.text_with_ws for token in sent_doc][start:end+1]).strip()
                            lemma_match += 1

                    
                    else:
                        id = e["id"]
                        en_trigger = trigger_en_pt_dict[id]["en"]
                        
                        pt_translations_lemma = trigger_en_pt_translations_lemma[en_trigger]
                        pt_translations_lemma = utils.normalizeTokens(pt_translations_lemma)
                        found = False
                        
                        for trans in pt_translations_lemma:
                            if re.search(r"\b"+re.escape(trans)+r'\b',sent_lemma):
                                translations_match +=1
                                trans_tokens_doc =nlp(trans)
                                trans_tokens = [token.text for token in trans_tokens_doc]
                                trigger_span = utils.contains(sent_lemma_tokens,trans_tokens,nlp)
                                if trigger_span:
                                    found = True
                                    start, end = trigger_span
                                    texts[ti]["golden-event-mentions"][ei]["trigger"]["previous_pt"] = texts[ti]["golden-event-mentions"][ei]["trigger"]["text_pt"]
                                    texts[ti]["golden-event-mentions"][ei]["trigger"]["text_pt"] = ''.join([token.text_with_ws for token in sent_doc][start:end+1]).strip()
                                    break

                        if not found:    
                            trigger_lemma = " ".join(trigger_lemma_tokens_aux)
                            if trigger_lemma in synonyms_br_lemma:
                                pt_syn_lemma = synonyms_br_lemma[trigger_lemma]

                                for syn in pt_syn_lemma:
                                    if re.search(r"\b"+re.escape(syn)+r'\b',sent_lemma):
                                        synonym_match +=1
                                        trans_sys_doc =nlp(syn)
                                        trans_sys_tokens = [token.text for token in trans_sys_doc]
                                        trigger_span = utils.contains(sent_lemma_tokens,trans_sys_tokens,nlp)
                                        if trigger_span:
                                                    found = True
                                                    start, end = trigger_span
                                                    texts[ti]["golden-event-mentions"][ei]["trigger"]["previous_pt"] = texts[ti]["golden-event-mentions"][ei]["trigger"]["text_pt"]
                                                    texts[ti]["golden-event-mentions"][ei]["trigger"]["text_pt"] = ''.join([token.text_with_ws for token in sent_doc][start:end+1]).strip()
                                                    
                        if not found:
                            texts[ti]["golden-event-mentions"][ei]["trigger"]["failed"] = -1
                            not_found +=1
            #print(ti)
        return texts
        




class WordAlignerMatch:
    def execute(self,texts,nlp):
        word_aligner = 0
        not_found = 0
        for ti, t in enumerate(texts):
            for ei, e in enumerate(t["golden-event-mentions"]):
                if "failed" in e["trigger"]:

                    en_trigger = texts[ti]["golden-event-mentions"][ei]["trigger"]["text"]
                    en_sent = texts[ti]["sentence"]
                    pt_sent = texts[ti]["sentence_pt"]
                    #print(en_sent,pt_sent,en_trigger)
                    res = Al.wordAligner(en_sent,pt_sent,en_trigger,nlp)
                    if not res or not Al.word_align_safe(res,en_trigger,nlp):
                        res = ""
                    if res:
                        word_aligner += 1
                    else:
                        not_found += 1
                    texts[ti]["golden-event-mentions"][ei]["trigger"]["t_aligned-bert"] = res
                    texts[ti]["golden-event-mentions"][ei]["trigger"]["previous_pt"]= texts[ti]["golden-event-mentions"][ei]["trigger"]["text_pt"]
                    texts[ti]["golden-event-mentions"][ei]["trigger"]["text_pt"]= res
            #print(ti)
        return texts