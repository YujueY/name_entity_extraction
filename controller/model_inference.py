# -*- coding: utf-8 -*- 
'''
@version V1.0.0
@Time : 2021/9/7 11:29 下午
@Author : azun
@File : model_inference.py 
'''
import regex as re
from controller.ner_label_predict import ner_clabel_predict
from controller import NER_LABEL_TYPE, message
from utils.time_cost import time_cost
from utils.logClass import logger
from controller import ner_map
from prettyprinter import pformat

nlp_logger = logger['nlp_logger']


class ModelInference():
    def __init__(self, model_path, device):
        self.ner_model = ner_clabel_predict(model_path=model_path,
                                            device=device)

    @time_cost
    def labels_inference(self, text: str, ner_label_dic: NER_LABEL_TYPE):
        '''
                按句号分割,需要提升效率，现在是一句一句推理的
        '''
        sentences = re.split(r"([。])", text)
        sentences.append("")
        sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]
        sentences_len = [len(x) for x in sentences]
        for sentence_idx, sentence in enumerate(sentences):
            res_ = self.ner_model.predict(sentence)
            # print(res_)
            # print('*' * 80)
            # print(sentence)
            # print(res)
            for res in res_:
                for x in res['entities']:
                    start = int(x[1])
                    end = int(x[2]) + 1
                    ner_obj = sentence[start:end]
                    doc_start = sum(sentences_len[:sentence_idx]) + start
                    doc_end = sum(sentences_len[:sentence_idx]) + end
                    assert sentence[start:end] == text[doc_start:doc_end]

                    entity_type = ner_map[x[0]]

                    ner_label_dic[entity_type].append(
                        {
                            'entity': ner_obj,
                            'start': doc_start,
                            'end': doc_end - 1,
                            'risk_level': 0,
                            'message': message[entity_type]
                        }
                    )
                    # doc_class = ner_label_class[x[0]]
                    # ner_label_dic[doc_class][x[0]]['items'].append(
                    #     {
                    #         'entity': ner_obj,
                    #         'start': doc_start,
                    #         'end': doc_end - 1
                    #     }
                    # )
        nlp_logger.info('*' * 40)
        nlp_logger.info('*' * 40)
        nlp_logger.info("ner_res")
        nlp_logger.info(pformat(ner_label_dic))
        nlp_logger.info('*' * 40)
        nlp_logger.info('*' * 40)
