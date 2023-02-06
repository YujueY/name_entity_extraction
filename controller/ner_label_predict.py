
# -*- coding: utf-8 -*-
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from BERT_NER_Pytorch.tools.common import init_logger, logger
from BERT_NER_Pytorch.models.transformers import WEIGHTS_NAME, BertConfig, AlbertConfig
from BERT_NER_Pytorch.models.bert_for_ner import BertSoftmaxForNer, BertCrfForNer
from BERT_NER_Pytorch.models.albert_for_ner import AlbertSoftmaxForNer, AlbertCrfForNer
from BERT_NER_Pytorch.processors.utils_ner import CNerTokenizer, get_entities
from BERT_NER_Pytorch.processors.ner_seq import convert_examples_to_features
from BERT_NER_Pytorch.processors.ner_seq import collate_fn
import copy

# from utils.msg import time_cost
logger.setLevel('WARNING')

MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertCrfForNer, CNerTokenizer),
    'albert': (AlbertConfig, BertCrfForNer, CNerTokenizer),
}


class ner_clabel_predict():
    def __init__(self, model_path='', num_labels=61, device=None):
        config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']
        if device == 'cpu':
            self.device = device
        else:
            self.device = 'cuda:' + str(device)
        self.model_path = model_path
        self.config = config_class.from_pretrained(self.model_path,
                                                   num_labels=num_labels,
                                                   loss_type='ce',
                                                   cache_dir=None, )
        self.model = model_class.from_pretrained(self.model_path, from_tf=bool(".ckpt" in self.model_path),
                                                 config=self.config, cache_dir=None)
        self.model.to(self.device)
        self.tokenizer = tokenizer_class.from_pretrained(self.model_path, do_lower_case=False)
        self.train_max_seq_length = 256
        self.model_type = 'bert'
        self.local_rank = -1

        self.id2label = {0: 'X', 1: 'B-Announcement_media', 2: 'B-Audit_reduction_amount', 3: 'B-Bid_bond',
                         4: 'B-Budget', 5: 'B-Construction_period', 6: 'B-Contract_standard_clause', 7: 'B-Deadline',
                         8: 'B-Delivery_time', 9: 'B-Number_of_service_providers', 10: 'B-Purchase_quantity',
                         11: 'B-Purchasing_content_service', 12: 'B-Qualifications_of_bidders',
                         13: 'B-Scoring_standard', 14: 'B-Service_period', 15: 'B-Technical_requirements',
                         16: 'B-Tender_time', 17: 'B-Warranty_Deposit', 18: 'B-project_name', 19: 'B-project_name_sub',
                         20: 'B-working_experience', 21: 'I-Announcement_media', 22: 'I-Audit_reduction_amount',
                         23: 'I-Bid_bond', 24: 'I-Budget', 25: 'I-Construction_period',
                         26: 'I-Contract_standard_clause', 27: 'I-Deadline', 28: 'I-Delivery_time',
                         29: 'I-Number_of_service_providers', 30: 'I-Purchase_quantity',
                         31: 'I-Purchasing_content_service', 32: 'I-Qualifications_of_bidders',
                         33: 'I-Scoring_standard', 34: 'I-Service_period', 35: 'I-Technical_requirements',
                         36: 'I-Tender_time', 37: 'I-Warranty_Deposit', 38: 'I-project_name', 39: 'I-project_name_sub',
                         40: 'I-working_experience', 41: 'S-Announcement_media', 42: 'S-Audit_reduction_amount',
                         43: 'S-Bid_bond', 44: 'S-Budget', 45: 'S-Construction_period',
                         46: 'S-Contract_standard_clause', 47: 'S-Deadline', 48: 'S-Delivery_time',
                         49: 'S-Number_of_service_providers', 50: 'S-Purchase_quantity',
                         51: 'S-Purchasing_content_service', 52: 'S-Qualifications_of_bidders',
                         53: 'S-Scoring_standard', 54: 'S-Service_period', 55: 'S-Technical_requirements',
                         56: 'S-Tender_time', 57: 'S-Warranty_Deposit', 58: 'S-project_name', 59: 'S-project_name_sub',
                         60: 'S-working_experience', 61: 'O', 62: '[START]', 63: '[END]'}
        self.id2label = {0: 'X', 1: 'B-Announcement_media', 2: 'B-Audit_reduction_amount', 3: 'B-Bid_bond',
                         4: 'B-Budget', 5: 'B-Construction_period', 6: 'B-Contract_standard_clause', 7: 'B-Deadline',
                         8: 'B-Delivery_time', 9: 'B-Number_of_service_providers', 10: 'B-Purchase_quantity',
                         11: 'B-Purchasing_content_service', 12: 'B-Qualifications_of_bidders',
                         13: 'B-Scoring_standard', 14: 'B-Service_period', 15: 'B-Technical_requirements',
                         16: 'B-Tender_time', 17: 'B-Warranty_Deposit', 18: 'B-project_name',
                         19: 'B-working_experience', 20: 'I-Announcement_media', 21: 'I-Audit_reduction_amount',
                         22: 'I-Bid_bond', 23: 'I-Budget', 24: 'I-Construction_period',
                         25: 'I-Contract_standard_clause', 26: 'I-Deadline', 27: 'I-Delivery_time',
                         28: 'I-Number_of_service_providers', 29: 'I-Purchase_quantity',
                         30: 'I-Purchasing_content_service', 31: 'I-Qualifications_of_bidders',
                         32: 'I-Scoring_standard', 33: 'I-Service_period', 34: 'I-Technical_requirements',
                         35: 'I-Tender_time', 36: 'I-Warranty_Deposit', 37: 'I-project_name',
                         38: 'I-working_experience', 39: 'S-Announcement_media', 40: 'S-Audit_reduction_amount',
                         41: 'S-Bid_bond', 42: 'S-Budget', 43: 'S-Construction_period',
                         44: 'S-Contract_standard_clause', 45: 'S-Deadline', 46: 'S-Delivery_time',
                         47: 'S-Number_of_service_providers', 48: 'S-Purchase_quantity',
                         49: 'S-Purchasing_content_service', 50: 'S-Qualifications_of_bidders',
                         51: 'S-Scoring_standard', 52: 'S-Service_period', 53: 'S-Technical_requirements',
                         54: 'S-Tender_time', 55: 'S-Warranty_Deposit', 56: 'S-project_name',
                         57: 'S-working_experience', 58: 'O', 59: '[START]', 60: '[END]'}
        self.markup = 'bios'
        self.label_list = list(self.id2label.values())

    # @time_cost
    def predict(self, text, prefix=""):

        test_dataset = self.load_and_cache_examples(text)
        # Note that DistributedSampler samples randomly
        test_sampler = SequentialSampler(test_dataset) if self.local_rank == -1 else DistributedSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1, collate_fn=collate_fn)
        # Eval!
        # logger.info("***** Running prediction %s *****", prefix)
        # logger.info("  Num examples = %d", len(test_dataset))
        # logger.info("  Batch size = %d", 1)

        results = []
        # output_submit_file = os.path.join(pred_output_dir, prefix, "test_prediction.json")
        # pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")
        for step, batch in enumerate(test_dataloader):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": None}
                if self.model_type != "distilbert":
                    # XLM and RoBERTa don"t use segment_ids
                    inputs["token_type_ids"] = (batch[2] if self.model_type in ["bert", "xlnet"] else None)
                outputs = self.model(**inputs)
            logits = outputs[0]
            preds = logits.detach().cpu().numpy()
            preds = np.argmax(preds, axis=2).tolist()
            preds = preds[0][1:-1]  # [CLS]XXXX[SEP]
            tags = [self.id2label[x] for x in preds]
            label_entities = get_entities(preds, self.id2label, self.markup)
            json_d = {}
            json_d['id'] = step
            json_d['tag_seq'] = " ".join(tags)
            json_d['entities'] = label_entities
            results.append(json_d)
            # pbar(step)

        # print(results)
        # logger.info("\n")
        return results
        # with open(output_submit_file, "w") as writer:
        #     for record in results:
        #         writer.write(json.dumps(record) + '\n')

    def load_and_cache_examples(self, text_a):

        class InputExample(object):
            """A single training/test example for token classification."""

            def __init__(self, guid, text_a, labels):
                """Constructs a InputExample.
                Args:
                    guid: Unique id for the example.
                    text_a: list. The words of the sequence.
                    labels: (Optional) list. The labels for each word of the sequence. This should be
                    specified for train and dev examples, but not for test examples.
                """
                self.guid = guid
                self.text_a = text_a
                self.labels = labels

            def __repr__(self):
                return str(self.to_json_string())

            def to_dict(self):
                """Serializes this instance to a Python dictionary."""
                output = copy.deepcopy(self.__dict__)
                return output

            def to_json_string(self):
                """Serializes this instance to a JSON string."""
                return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

        examples = [InputExample(guid='test-1', text_a=list(text_a), labels=['O'] * len(text_a))]
        # print(self.tokenizer.pad_token)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=self.tokenizer,
                                                label_list=self.label_list,
                                                max_seq_length=self.train_max_seq_length,
                                                cls_token_at_end=bool(self.model_type in ["xlnet"]),
                                                pad_on_left=bool(self.model_type in ['xlnet']),
                                                cls_token=self.tokenizer.cls_token,
                                                cls_token_segment_id=2 if self.model_type in ["xlnet"] else 0,
                                                sep_token=self.tokenizer.sep_token,
                                                # pad on the left for xlnet
                                                # pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                                                pad_token=
                                                self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if self.model_type in ['xlnet'] else 0,
                                                )
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)
        return dataset


if __name__ == "__main__":
    # ner_softmax_predict=ner_clabel_predict('/szzn/azun/document_zaozhuang_tobacco/data/model_file/checkpoint-3584',device='cpu')
    ner_softmax_predict = ner_clabel_predict(
        '/app_name/nlp/azun/supertext_auto/data/exp/model/yanyi-zaozhuangbiaoshu/chinese_roberta_L-4_H-512_softmax_枣庄标书+_v4', device='cpu')
    # txt_path='/var/www/html/brat/data/policy_v1_label/0.txt'
    # texts=[]
    # with open(txt_path,'r') as f:
    #     for line in f.readlines():
    #         texts.append(line.strip())

    # texts = ['新引进的市外资“大好高”项目']
    # texts = [
    #     '山东枣庄烟草有限公司知识产权代理机构服务项目,招标文件招标编号：SDDY-2020-招标人：山东枣庄烟草有限公司招标代理机构：山东鼎益建设项目管理有限公司日期：2020年11月目录第一章招标公告一、项目名称：山东枣庄烟草有限公司知识产权代理机构服务项目二、项目编号：三、招标方式：公开招标四、项目内容：详见招标文件五、招标控制价：/万元。六、投标人资格要求：1、在中华人民共和国境内注册，具有独立法人资并依法取得企业营业执照且营业执照处于有效期；2、具有较强的经济实力、良好的商业信誉和健全的财务会计制度，并在人员、设备、资金等方面具有相应的服务能力；3、遵守《中华人民共和国招标投标法》及相关法律、法规和规章；4、本项目不接受联合体投标。']
    # texts = [
    #     '服务总价款及付款方式：（以下价款均为含税价格，税率6%）本合同项下乙方提供的服务总价款为RMB350650元（大写：人民币叁拾伍万零陆佰伍拾圆整）；付款方式：预付款：甲方应于合同生效后7日内以汇款或即期支票（同城）方式向乙方支付合同价款总额的20%作为预付款，共计：RMB70130元（大写：人民币柒万零壹佰叁拾圆整）。甲方已支付的预付款可抵作本合同中的软件安装费、技转培训费、项目管理费、软件实施费，本合同各服务项目及服务阶段之间各自独立，乙方凭对应交付文档分别抵用预付款或向甲方请款，甲方在上述预付款抵用完毕前无需向乙方再次付费，在预付款抵用完毕后，甲方需向乙方按时支付费用。']
    # texts = [
    #     '山东枣庄烟草有限公司台儿庄区营销部办公楼电路维修改造项目招标文件MINGZHENG招标人：山东枣庄烟草有限公司代理机构：山东明正招标咨询有限公司项目编号：SDMZ-2020-067日期：二〇二〇年十月目录第十四章合同-40-第一章招标公告山东明正招标咨询有限公司受托，就山东枣庄烟草有限公司台儿庄区营销部办公楼电路维修改造项目以公开招标的方式组织招标。']
    # texts = [
    #     '山东枣庄烟草有限公司印制价格标签项目,招标文件项目编号：SDJY-2021-招标人：山东枣庄烟草有限公司招标代理机构：山东建业招标咨询有限公司二零二一年三月第一章招标公告................................................1第二章投标人须知..............................................3第三章评标办法...............................................27第四章合同条款和格式.........................................30第五章项目说明...............................................32第六章投标文件格式...........................................34第一章招标公告一、项目名称：山东枣庄烟草有限公司印制价格标签项目二、项目编号：SDJY-2021-三、招标方式：公开招标四、采购内容情况：内容投标人资格预算金额1.投标人']
    #
    # a = {
    #     "text": "山东枣庄烟草有限公司峄城区营销部更换雨棚防火岩棉项目招标文件项目编号：SDXY-2019-025招标人：山东枣庄烟草有限公司招标代理机构：山东信一项目管理有限公司二〇一九年十月目录第一章招标公告一、项目名称：山东枣庄烟草有限公司峄城区营销部更换雨棚防火岩棉项目二、项目编号：SDXY-2019-025三、招标方式：公开招标四、采购内容情况：本次招标采用资格后审方式。五、获取招标文件时间、地点及方式：时间：2019年10月日至2019年10月日（法定节假日除外），上午9:00-11：30、下午2:00-4:00（北京时间）。地点：枣庄市高新区枣庄国际大厦15楼方式：购买招标文件时请携带原件及复印件加盖公章：①营业执照（副本）、②税务登记证（副本）、③组织机构代码证（副本）或三证合一营业执照（副本）、④法定代表人证明或法定代表人授权委托书（含法定代表人身份证复印件）原件及相应的身份证原件参加报名，不能证明其有效资格的投标人，招标人将予以拒绝接收其报名。投标报名时的资格查验不代表资格审查最终通过或合格，投标人最终资格确认以招标现场组织的评标委员会的资格后审结果为准。",
    #     "file_info": "v1-----1010011",
    #     "label": {"project_name": {"山东枣庄烟草有限公司峄城区营销部更换雨棚防火岩棉项目": [[0, 25]]}, "Tender_time": {"二〇一九年十月": [[81, 87]]}}}
    # texts = [a['text']]
    # texts = [
    #     '山东枣庄烟草有限公司峄城区营销部更换雨棚防火岩棉项目招标文件项目编号：SDXY-2019-025招标人：山东枣庄烟草有限公司招标代理机构：山东信一项目管理有限公司二〇一九年十月目录第一章招标公告..................................................2第二章投标人须知................................................5第三章评标办法........................']
    texts = ["山东枣庄烟草有限公司标准服务站维修项目招标文件MINGZHENG招标人：山东枣庄烟草有限公司代理机构：山东明正招标咨询有限公司项目编号：SDMZ-2020-010日期：二〇二〇年三月目录第一章招标公告山东明正招标咨询有限公司受托，就山东枣庄烟草有限公司标准服务站维修项目以公开招标的方式组织招标。欢迎符合条件的投标人参加投标。一、招标人名称：山东枣庄烟草有限公司二、招标代理机构名称：山东明正招标咨询有限公司三、项目名称及项目编号：项目名称：山东枣庄烟草有限公司标准服务站维修项目项目编号：SDMZ-2020-010四、招标内容：本项目为山东枣庄烟草有限公司标准服务站维修项目工程,具体内容详见工程量清单。五、投标单位要求：1、在中华人民共和国境内合法注册，具备企业法人营业执照及相应的经营范围；2、具有履行合同的能力，有良好的商业信誉和健全的财务会计制度；3、投标单位须具备建筑工程施工总承包或建筑装饰装修工程专业承包三级及以上资质，且取得安全生产许可证，项目经理为贰级及以上注册建造师并具有安全生产考核合格证书；4、遵守《中华人民共和国招标投标法》及相关法律、法规和规章；5、本项目不接受联合体投标。六、采购文件发售时间：2020年3月日—2020年月日每日08：30时—17：00时；（节假日除外，过时不予办理）发售地点：山东明正招标咨询有限公司（枣庄市新城区嘉汇大厦2-C-12号）购买招标文件时须携带以下证件原件及加盖单位公章的复印件办理（原件审查，复印件留存）：营业执照、资质证书、安全生产许可证、项目经理注册建造师证书、安全生产考核合格证书、法定代表人证明或法人授权委托书。投标人报名时的资格查验不代表资格审查最终通过或合格，最终资格的确认以开标现场组织的资格后审结果为准。采购文件售价：300元/份，售后不退。七、投标截止暨开标时间：2020年月日日9:30时开标地点：枣庄市新城区嘉汇大厦2-C-23号会议室八、本项目联系人及联系电话：联系人：张经理联系电话：0632-3191678第二章投标人须知投标人应仔细阅读本招标文件的所有内容（包括答疑、补充、澄清以及修改等），按照招标文件要求以及格式编制投标文件，并保证其真实性，否则其一切后果自负。一、投标人须知前附表二、当事人1“招标人”系指山东枣庄烟草有限公司。2“投标人”系指响应招标并且符合招标文件规定的资格条件和参加投标的企业法人或事业法人。3“评标委员会”系指根据《中华人民共和国招标投标法》等相关法律法规以及规定，组成以确定中标人的临时组织。4“中标人”系指由评标委员会综合评审确定的对招标文件做出实质性响应，综合竞争实力最优，取得与招标人签订合同资格的投标人。5“招标代理机构”系指山东明正招标咨询有限公司。三、招标依据及原则1《中华人民共和国建筑法》；2《中华人民共和国招标投标法》；3《中华人民共和国合同法》；4《中华人民共和国招标投标法实施条例》；5《建设工程质量管理条例》；6其他有关法律、行政法规以及省、市规范性文件规定。四、项目概况1项目名称：山东枣庄烟草有限公司标准服务站维修项目；2建设地点：枣庄烟草有限公司滕州市级索、龙阳、羊庄服务站；3资金来源：自筹。五、合格的投标人1基本条件：1.1具备招标公告所要求的资格、资质；1.2提供的资格、资质证明文件真实有效；1.3向招标代理机构购买了招标文件并登记备案；1.4向招标代理机构交纳了投标保证金；1.5在以往的政府采购活动中没有违纪、违规、违约等不良行为；1.6对招标文件做出实质性响应；2遵守《中华人民共和国招标投标法》、《中华人民共和国招标投标法实施条例》及其它有关法律法规的规定。3投标人财务状况及信誉良好，没有处于被责令停业，投标资格未被取消，财产未被接管、冻结，未处于破产状态。4在最近三年以来没有骗取中标和严重违约及重大工程质量问题。如有，招标人将有权拒绝其投标。5投标人提供的证明材料内容必须真实可靠。符合上述条件的投标人即为合格的投标人，并将具有参与评审的资格。六、保密参与招标投标活动的各方应对招标文件和投标文件中的商业和技术等秘密保密，违者应对由此造成的后果承担法律责任。七、语言文字以及度量衡单位1所有文件使用的语言文字为中文。专用术语使用外文的，应附有中文注释；2所有计量均采用中华人民共和国法定的计量单位；3所有报价一律用人民币，货币单位：元。八、勘查现场1自行踏勘现场，投标人承担踏勘现场所发生的自身费用。2招标人向投标人提供的有关现场的资料和数据，是招标人现有的能使投标人利用的资料，招标人对投标人由此而做出的推论、理解和结论概不负责。3投标人经过招标人允许，可以进入项目现场踏勘，但不得因此使招标人承担有关责任和蒙受损失。投标人应对踏勘现场而造成的死亡、人身伤害、财产损失、损害以及其它任何损失、损害和引起的费用和开支承担责任。九、投标答疑1投标人对招标文件、踏勘现场有疑问或者询问，需招标人解答或者答疑时，应于2020年月日17:00时前，以加盖投标人单位公章的书面文件提出，采用信函、传真或者直接送达的形式通知招标代理机构，同时将电子版文件以电子邮件的形式发送至sdmz999@163.com。招标人将对投标人提出的所有疑问或者询问进行综合答复，解答或者答疑内容应在招标文件规定范围内，不得对招标文件实质性条款进行改动，并形成书面文件报招标单位审查备案后，统一用电子邮箱发给所有招标文件收受人，并电话通知。2投标人不在规定时间提出疑问或者询问，视为认同招标文件以及答疑文件内的所有要求，投标人不按照招标文件、解答或者答疑要求投标的，后果自负。3投标答疑时间：2020年月日。十、偏离:只允许合理正偏离。十一、施工要求1本工程应按国家建设部、工程施工技术（验收）规程、规范、要求标准施工。2严格按照施工图及有关技术要求、文件、资料进行施工。十二、履约担保无。十三、采购代理服务费1采购代理服务费收取按照国家计委《招标代理服务收费管理暂行办法》（计价格〔2002〕1980号）、国家发展改革委办公厅《关于招标代理服务收费有关问题的通知》（发改办价格〔2003〕857号）,以中标金额为基准计算并收取，此费用由中标人支付。十四、费用承担投标人准备和参加投标活动发生的一切费用自理。十五、其他条款1投标人中标后直至验收止，未经招标人同意，中标人不得以任何形式和理由转包或者分包；如出现上述情形，招标人可取消其中标资格，并与其立即解除合同，由此引起的经济损失全部由中标人承担。2不论招标过程和结果如何，投标人的投标文件均不退还；3招标文件中的实质性条款，各投标人必须按照招标文件的要求作出实质性响应，否则按废标处理。4除非有特殊要求，招标文件不单独提供项目所在地的自然环境、气候条件、公用设施等情况，投标人被视为熟悉上述与履行合同有关的一切情况。第三章招标文件一、招标文件的构成招标文件是用以阐明工程施工、招标程序和合同格式的规范性文件，是招投标双方首要约束性文件。招标文件主要由以下部分组成：1招标公告；2投标人须知；3招标文件内容及要求；4投标报价、投标文件编制以及投标保证金；5投标人应当提交的资格、资信证明文件；6投标截止时间、开标时间以及地点；7开标、评标、中标以及废标；8合同的授予、合同主要条款；9工程量清单；10技术标准和要求；11评标办法；12投标文件格式。招标人或者招标代理机构对招标文件所作的答疑、澄清或者修改，作为招标文件的组成部分。二、招标文件的澄清1投标人获得招标文件后，应仔细检查招标文件是否齐全。如有残缺、遗漏或者不清楚的，应在得到招标文件后一日内，以加盖投标人单位公章的书面文件提出，采用信函、传真或者直接送达的形式通知招标代理机构，同时将电子版文件以电子邮件的形式发送至sdmz999@163.com，否则，由此引起的损失由投标人自负。同时，投标人有义务对招标文件的准确性进行复核，如发现有任何错误（打印的错误、逻辑的错误）或者前后矛盾的，应在规定提交答疑的时间内提交给招标代理机构，否则，投标人应无条件接受招标文件所有条款。2招标人对已发出的招标文件进行必要澄清（包括补充）的，应当在招标文件要求提交投标文件截止时间十五日前，以书面形式通知所有招标文件收受人，但不指明澄清问题的来源。3招标文件的澄清（包括补充）在同一内容的表述上不一致时，以最后发出的为准。4投标人认为招标文件存在歧视性条款或者不合理要求等需要澄清的，应在规定时间内一次性全部提出。在规定时间未一次性提出或者对已澄清的条款再提异议者，即视为同意和接受相关条款。三、招标文件的修改招标人对已发出的招标文件进行必要修改的，在招标文件要求提交投标文件截止时间十五日前，报招标单位批准发布变更或者更正公告，并以书面并附带电子邮件形式通知所有招标文件收受人。第四章投标报价、投标文件编制以及投标保证金一、报价依据投标人应按照招标人提供的招标文件、设计方案、技术资料等，根据施工现场实际情况，结合投标人自身技术、管理水平、经营状况、机械配备，按照企业定额和市场价格信息及有关规定进行报价。投标报价1工程量清单是招标文件的一部分，投标报价中应综合工程量清单特征描述、工程量计算规范、设计图纸、招标文件及规范规程等需包含的所有内容，并完成与之相关的各种主要及辅助工作。在合同实施期间，不因工程量清单子项项目特征和工程内容是否描述完整而调整综合单价。2工程量清单报价表应与招标文件(含补遗书、答疑纪要)中投标须知、合同条款、工程技术要求、法律、法规、条例、规定及规范、设计图等文件结合理解和使用。3凡参加本项目投标的投标人均被认为是有经验、有实力的承包商，其完全有能力根据项目进度状况和要求聚集、调动、分配各种资源以供本项目使用。4投标报价应由投标人依据上述工程量清单、招标图、地勘资料、进度要求、施工现场实际情况（投标人应充分了解施工现场所能提供的施工作业面、地下管线可能造成的工作难度、以及周边建、构筑物的保护和需提供的必要通道等)、当地气候条件，结合投标人自身技术和管理水平、经营状况、机械配备(其中主要机械设备应为投标人所有，且该机械设备是本项目随时可以使用的)以及编制的施工组织设计和招标文件的有关要求，参照建设行政主管部门发布的计价定额或企业定额以及本说明自主编制确定。5投标人的投标报价应是工程量清单及招标文件所确定的招标范围内的全部工作内容的价格体现，应包括本工程的分部分项工程量清单费用、措施项目清单费用、其他项目清单费用、规费、税金等完成本工程及达到保修期限所需的一切费用及风险（即本工程的直接费、间接费、管理费、代理费、利润、风险金、保险金、材料保管及场内运输费、按设计要求和验收规范及政府相关管理部门要求施工单位应进行的测试费用、规费、税金及政策性文件规定的各项费用等。此外，投标人应根据企业自身情况考虑材料损耗率，在投标报价时将材料损耗综合考虑在所报综合单价内，中标后不作调整。综合单价为完成一个规定清单项目所需的人工费、材料和工程设备费、施工机具使用费和企业管理费、利润规费、税金以及合同约定范围内的风险费用。6投标人可自行到工地踏勘以充分了解工地位置、情况、道路、储存空间、装卸限制及任何其他足以影响报价的情况，任何因忽视或误解工地情况而导致的索赔或工期延长申请将不被批准。7投标文件中工程量清单的投标总价应与投标书的投标价保持一致，对于出现两个投标价的投标文件，招标人不予接受。本次招标如果设计图纸、技术措施表与工程量清单有矛盾时，以及招标文件与本清单有矛盾的地方，不管差异多大，投标人在答疑时均以书面提出，否则，投标人的投标报价被认为已包含这部分差异（如有）造成的费用，中标后均以发包人的解释和要求执行且投标报价不作调整。8投标人应充分考虑施工现场周边设施及其他设施的保护处理。如在保护处理过程中对相关设施造成损坏，由承包人按招标人相关部门要求及时修复和赔偿。以上状况所产生的各项费用应在报价时考虑，招标人不再另行支付(经发包人认可的特殊情况除外)。9为便于评标及投标人报价，就清单中部分项目应综合的主要内容加以了陈述供投标时参考，投标人不要产生“被陈述的项目为该项目只能综合的内容”。未被陈述项目应综合的内容均不能作为向发包方索赔的依据，投标人应依据相关资料综合其它的内容进行报价。10施工用水、用电的具体接入点的位置在现场踏勘时由招标人指定。投标人踏勘时应充分进行现场考察，包括已形成的外部公共道路至施工场地道路、施工场地内部道路、临时用水用电从接驳点到使用位置的费用及排污费用均由承包人自行完成施工、办理相关手续并承担相应费用，并应包含在报价中。投标人应在投标报价中综合考虑现场施工用水、施工用电条件的不足导致的施工时可能增加的相关费用，中标后不作调整。11工程量清单中的每一个细目，都必须填入单价及合价。清单中未填入合价的细目，不能得到结算与支付。12投标人报价时应按《建设工程工程量清单计价规范》要求提供详细的“工程量清单综合单价分析表”“规费、税金项目清单与计价表”等。13为避免不平衡报价，投标人在同一标段内，相同清单项目综合单价或相同品质要求的材料单价应保持一致。若出现不一致的情况，招标人经分析后确属不正常的，有权按照较低的单价进行结算与支付。14投标人应充分熟悉图纸，了解项目周边的情况与条件，结合本项目的特殊性、复杂程度、工期情况及招标文件要求，以及与各专业公司作业的交叉影响，确保项目按期保质完成、符合各部门管理要求等而做出的综合报价。15由于施工工艺、施工机械、组织措施、技术措施等的变化，均不调整投标单价。16工程竣工验收前的通风检测、空气检测等检测费用投标人应综合考虑在投标报价中，结算时不再调整。17工程施工过程中因施工便利增加的工程或措施费用由承包人自行承担，发包人特殊指令的除外。18招标人未提供材料、设备品牌及规格型号的，投标人须注明品牌及规格型号(砂、石等无品牌的除外)。19若投标人的材料价格表中的材料价格与分部分项工程量清单综合单价分析表中的相应材料价格不一致时，以两者较低的材料价格为准。20施工过程中，投标人必须按招标文件中所给品牌及规格型号执行，若发包人发现承包人所用品牌与投标文件所报品牌不一致时应令其立即改正，造成的费用损失及工期延误由承包人自行承担。21施工过程中若发生承包人所用品牌停产无法供货或其他原因等造成不能按其投标时约定的品牌及厂家供货的情况，承包人必须在招标人提供的指定备选品牌、型号中选用，不得采用其他品牌及型号(经发包人认可的同档次的其他品牌及型号除外)，且不得调整中标材料价格及综合单价。22投标人选择的材料必须满足国家及地区相关规范和标准要求。主要材料应包括(但不仅限于)三材、地材、装饰材料、安装材料及设备等。23.本次招标工程工程量清单所列项目及工程量是投标人作为投标的共同基础，各投标单位不得更改。招标人有权取消或增加清单项目，无论取消或增加的项目多少、金额大小，其它项目的单价均不作调整，且不因增减项目做任何补偿。取消的清单项目不进行工程计量和支付，增加的项目按合同约定进行处理。24.所有材料均须符合现行国家标准或行业标准及设计要求。25.本项目设控制价，超过控制价的投标为无效报价。三、投标文件的签署要求1投标文件用不褪色的材料书写或者打印。招标文件要求投标人法定代表人或者被授权代表签字或签章处，均须本人用黑色中性签字笔签署（包括姓和名）并加盖单位公章，不得用签名章、签字章等代替，也不得由他人代签，否则其投标无效。被授权代表人签字的，投标文件应附法定代表人授权委托书，否则其投标无效。2“投标函”、“法定代表人授权委托书”必须由法定代表人签署，否则其投标无效。四、投标文件的盖章要求投标人在投标文件以及相关书面文件中的单位盖章（包括印章、公章等）均指与投标人名称全称相一致的标准公章，不得使用其他形式（如带有“专用章”、“合同章”、“财务章”、“业务章”等）的印章；否则，评标委员会有权确定投标无效。五、投标文件的时间单位、有效期以及费用1除招标文件中另有规定外，投标文件所使用的“天”、“日”均指日历天。2投标有效期为90日历天，自投标截止之日起90日历天内投标文件以及其补充、承诺等部分均保持有效。在招标文件规定的投标文件有效期满之前，如果出现特殊情况，招标代理机构可在投标有效期内要求投标人延长有效期，要求与答复均以书面通知为准并作为招标文件和投标文件的组成部分；投标人可以拒绝上述要求而其投标保证金不被没收，拒绝延长投标文件有效期的，其投标失效；同意上述要求的，既不能要求也不允许其修改投标文件，有关退还和没收投标保证金的规定在投标有效期的延长期内继续有效。3投标人应自行承担其准备和参加投标活动发生的所有费用。不论投标结果如何，招标人或者招标代理机构不承担任何费用。六、投标文件格式以及编制要求根据有关规定，投标文件按照以下要求、格式统一编制：1封面设置。投标文件材料封面设置包括：投标文件、项目名称、项目编号、投标单位全称和投标文件完成时间。投标单位全称填写“×××公司”。2投标文件内容。投标人应按照招标文件的要求以及格式编写投标文件；对招标文件要求填写的表格或者资料不得缺少或者留空。2.1投标文件不得加行、涂改、插字或者删除。2.2投标人须按照招标文件中的投标文件格式要求逐项填写。3投标文件商务标装订。投标文件的正本与副本应分别胶装成册；每份投标文件以A4纸张制作，并编制目录，目录、内容标注连续页码。否则，招标代理机构不予受理。4投标人可对施工现场以及其范围环境进行考察，以获取有关编制投标文件和签署实施合同所需的各项资料，投标人应承担现场考察的费用、责任和风险。5投标人编制投标文件时，若有偏离之处，请如实在商务或者技术偏离表中注明。6投标人不得递交备选投标方案。7投标文件数量以及要求。投标人应准备五套纸质投标文件。五套纸质商务文件必须完全一致。技术标须与商务标单独密封；未按规定编制、签署、制作、装订和密封的投标文件，其投标无效，技术标书不分正副本。七、投标文件的组成投标人应按照招标文件的要求以及格式编制投标文件，并保证其真实性、准确性以及完整性，并按照招标文件要求提交全部资料并做出实质性响应，否则其投标无效。1投标文件的组成投标文件由包括报价文件、技术文件、商务文件共三部分组成：2报价文件2.1开标一览表；开标一览表一式三份单独密封，以便唱价时使用。3技术文件3.1分部分项工程的主要施工方案；3.2工程投入的主要施工机械设备情况、主要施工机械进场计划；3.3劳动力安排计划；3.4确保工程质量措施；3.5确保安全生产措施；3.6确保文明施工措施；3.7确保工期的技术组织措施；3.8主要材料、设备进场计划；3.9工程总进度图表；3.10施工平面布置图；3技术文件为匿名标书，技术标“暗标”部分的制作和装订的具体要求：技术标书为匿名标书，标书封面采用统一提供的封面格式（见后附格式），封面不得出现页眉、页脚、页码，用A3标准白色复印纸复印后胶装（封面以前面为准，剩多少放在后面）；技术标书内容用纸（施工进度图及施工现场平面图除外）均采用A4标准白色复印纸打印，标题为宋体加黑三号字，内容使用宋体四号字标准字间距；行距：固定值22磅；页边距设置：上1.9㎝、下1.9cm、左2.2cm、右1.9cm，黑色打印，不得出现彩色内容，不得出现手写、不得出现页码、不得出现页眉页脚，不得作任何标记。如不响应上述要求或出现投标企业名称、页码、标志性图案及能识别该单位标记等不按要求制作的技术标书，则技术分为零分。4商务文件4.1投标函；4.2法定代表人身份证明书及投标文件签署授权委托书；4.3投标报价说明、投标总价、造价汇总表、分部分项工程量清单计价表、措施项目清单计价表、其他项目清单计价表、零星工作项目计价表、分部分项工程量清单综合单价分析表、措施项目费分析表、材料价格表等及投标报价需要的其他材料。4.4项目经理及项目班子的人员配备；且中标后，项目管理人员未经招标人同意不得替换；4.5投标人的企业业绩（合同在商务标书中装订复印件并加盖公章）；4.6其它需说明的内容。八、投标保证金以及退还1投标保证金交纳金额投标人在递交投标文件前，须缴纳投标保证金：伍仟元整，且必须在2020年月日16:00时前到账，否则不予接受，视为放弃投标。2投标保证金以银行电汇的形式交纳。3保证金的退还3.1投标人在招标文件要求提交投标文件截止时间前书面要求撤回投标文件的，招标人或者招标代理机构自收到投标人书面撤回文件之日起5日内退还已收取的投标保证金。3.2招标代理机构在中标通知发出后五个工作日内退还未中标人的投标保证金，在合同签订并备案后五个工作日内退还中标人的投标保证金；4没收投标保证金发生下列情况之一，投标保证金将被招标人没收。4.1提供的有关资料不真实或者提供虚假材料的；4.2参与公开招标后投标人撤回报价或者退出公开招标活动的；4.3开标后投标人撤回全部或者部分投标文件的；4.4损害招标人或者招标代理机构合法权益的；4.5投标人向招标代理机构、招标人、专家提供不正当利益的；4.6经评标委员会认定有故意哄抬报价、串标或者其它违法行为的;4.7法律、行政法规以及有关规定的其它情形。第五章投标人应当提交的资格、资信证明文件一、投标人在投标截止时间前须提交的商务资格、技术支持等证明材料：1需提供证明材料：上述证明材料中第.4.7项是投标人开标时必须提供的资格审查材料，届时未提供或者提供不全的，其投标无效。以上所有证明材料原件的复印件加盖单位公章后装入商务投标文件中，不装、漏装按无效标处理。二、相关规定1投标人的资格证明材料必须真实、有效、完整，且以中文为准，其中的字体、印章要清晰，否则其投标无效。2超出营业执照经营范围的，其投标无效。营业执照副本等原件在年检期间或者无法提供的，可提供由发证机关出具证明材料原件或者由公证机关出具的公证件原件，否则其投标无效。第六章投标截止时间、开标时间以及地点一、投标文件递交以及截止时间1投标人应当在招标文件要求提交投标文件截止时间前，将投标文件按要求密封送达投标地点。在招标文件要求提交投标文件的截止时间后送达的投标文件，招标代理机构不予受理。2投标人可对现场工作人员的资格和递交投标文件截止时间进行监督。如有异议，应以书面形式并签署单位名称以及法定代表人或者被授权代表姓名后，在投标文件开启前递交至招标单位现场监督人员，以便及时处理。否则视为同意和接受。3投标文件的递交截止时间：2020年月日日9时30分止。二、投标文件的密封和标记投标人提交的投标文件分为报价、技术、商务三部分，分别加以密封。封套上标明招标项目编号、项目名称、及投标人名称等，在封签处标注“请勿在2020年月日日9:30时之前启封”字样（见附件），并加盖投标单位公章、法定代表人或被授权代表签字或盖章，无分装、无密封、无盖章，投标无效。三、投标文件的修改与撤回1投标人在招标文件要求提交投标文件截止时间前，可以补充、修改、替代或者撤回已提交的投标文件，并书面形式通知招标人或者招标代理机构。补充、修改的内容为投标文件的组成部分。2在提交投标文件截止时间后到招标文件规定的投标有效期终止之前，投标人不得补充、修改、替代或者撤回其投标文件。投标人撤回的，其投标保证金将被没收。四、开标地点枣庄市新城区嘉汇大厦2-C-23号会议室。五、开标时间开标时间：2020年月日日9时30分。第七章开标、评标、中标以及废标一、开标程序1宣布开标纪律；2公布在投标截止时间前递交投标文件的投标人名称；3宣布主持人、唱标人、记录人等有关人员姓名；4投标人相互检查投标文件密封情况，并签字确认；5按照投标人报名顺序，宣布投标文件开启顺序；6按照顺序当众开标，公布投标人名称、投标保证金的递交情况、投标报价、工期等内容，并记录在案；7投标人法定代表人（或者被授权代表）、记录人等有关人员在开标记录上签字确认；8开标结束。二、开标1参加招标会的代表必须签名报到，法定代表人或被授权代表出具授权委托书和身份证以证明其出席，否则不予接收投标文件。2检查投标文件密封情况，由投标人法定代表人或者被授权代表互相检查各投标人投标文件的密封情况，并请各投标人法定代表人或者被授权代表签字确认。投标人法定代表人或者被授权代表认为某个或者某些投标人的投标文件密封不符合规定的，应当面提出，招标代理机构现场记录，相关各方投标人法定代表人或者被授权代表签字确认后，报招标单位现场监督人员和评标委员会处理。经确认无异议的，由招标代理机构工作人员当众拆封，开启各投标人投标文件。按照上述规定开启投标文件后，投标人再对投标文件的密封情况提出异议的，招标人或者招标代理机构不予受理。3由招标代理机构工作人员唱标。3.1唱标顺序：按照投标人投标签到顺序进行。3.2唱标内容：唱标人当众宣读投标人名称、投标报价、投标文件的其他主要内容，投标人若有报价和优惠未被唱出，应在开标时以及时声明或者提出，否则招标代理机构对此不承担任何责任。4投标文件有下列情况之一，招标代理机构不予受理：4.1逾期送达的或者未送达指定地点的；4.2未按照招标文件要求密封、标记的；4.3违反招标、投标纪律的；4.4开启投标文件后，投标人再对投标文件的密封情况提出异议的。5开标和唱标由招标代理机构指定专人负责唱标和记录，开标记录由投标人法定代表人或者被授权代表、记录人等有关人员签字确认。6投标人对开标有异议的，应当在开标现场以书面形式提出，招标人或者招标代理机构应当场给予答复，并制作记录，投标人法定代表人或者被授权代表、招标人代表、招标代理机构签字确认。7参加开标会议的招标人代表不得参加评审；开标记录由代理机构保存，在商务打分结束后提交评标委员会审核。三、评标委员会1评标委员会的组成评标由依法组建的评标委员会负责。评标委员会由评标专家组成，成员人数为五人，其中技术、经济等方面的专家不得少于成员总数的三分之二。2评审专家的抽取2.1由招标代理机构采用随机抽取方式，从专家库中确定评标委员会成员。3评标委员会负责对各投标文件进行评审、比较、评定，并确定中标人。4招标单位代表参加开标会议四、评标程序1宣布评标纪律以及回避提示；2组织推荐评标委员会组长；3资格性审查；4符合性审查；5技术评审；6澄清有关问题；7比较与评价；8确定中标人；9编写评标报告；10宣布评标结果。五、评标1评审1.1投标文件初审。初审分为资格性检查和符合性检查。资格性检查：依据法律法规和招标文件的规定，对投标文件的资格证明、投标保证金等进行审查，以确定投标人是否具备投标资格。符合性检查：依据招标文件的规定，从投标文件的有效性、完整性和对招标文件的响应程度进行审查，以确定是否对招标文件的实质性要求做出响应。1.2澄清有关问题。评标委员会可以书面方式，要求投标人对投标文件中含义不明确、对同类问题表述不一致或有明显文字和计算错误的内容作必要的澄清、说明或补正。评标委员会不得向投标人提出带有暗示性或诱导性的问题，或向其明确投标文件中的遗漏和错误。投标人的澄清、说明或者补正应当采用书面形式，由其授权的代表签字，并不得超出投标文件的范围或者改变投标文件的实质性内容。投标文件不响应招标文件的实质性要求和条件的，评标委员会予以拒绝，并且不允许投标人通过修正或撤销其不符合要求的差异或保留，使之成为具有响应性的投标。1.3投标文件计算错误的修正。评标委员会在对实质上响应招标文件要求的投标文件进行评估时，除招标文件另有约定外，须进行修正的，应符合下列原则：报价部分以正本为准。投标文件中《报价一览表》内容与分项报价、明细表内容不一致的，以《报价一览表》为准。大写金额和小写金额不一致的，以大写金额为准；总价金额与按照单价汇总金额不一致的，以总价为准；单价金额小数点有明显错位的，应以总价为准，并修改单价，总价与唱标单不一致的，以唱标单为准；对不同文字文本投标文件的解释发生异议的，以中文文本为准；用数字表示的数额与用文字表示的数额不一致时，以文字数额为准。单价与工程量的乘积与总价之间不一致时，以总价为准。按前款规定调整后的报价，经投标人确认后产生约束力。1.4比较与评价。按照招标文件规定的评分方法和标准，对资格性检查和符合性检查合格的投标文件进行商务和技术评估，综合比较与评价。1.5投标文件中没有列入的价格和优惠条件在评标时不予考虑。2投标文件有下列情形之一的，评标委员会按废标处理：2.1投标文件未按要求加盖单位公章、法定代表人或被其授权代表签字的；2.2投标文件未按本招标文件规定制作或未响应招标文件的要求的；2.3投标文件未按规定的格式填写，内容不全或关键字迹模糊、无法辨认的；2.4投标人针对同一项目递交两份或多份内容不同的投标文件，或在一份投标文件中对同一招标项目报有两个或多个报价，且未声明哪一个有效的；2.5投标人未按照招标文件的要求提交投标保证金的；2.6投标人法定代表人或委托代理人未按时参加开标会议或参加开标会未提供有效证明的；2.7营业执照副本、资质证书副本、安全生产许可证等资格证明材料原件未提供或提供不齐全者；2.8开标时投标人未提供注册建造师证书原件；2.9所提供有关证书、证明原件涂改、转让或提供虚假材料的；2.10投标企业清单报价工程量与本招标文件所附清单工程量不一致的；2.11投标报价高于招标控制价的；2.12未提供所要求的资格证明材料原件或所提供资格证明材料原件不齐全的；2.13未提供报价一览表的；2.14有下列情形之一的，由评标委员会进行核查，经集体表决，按照少数服从多数的原则，认定为串通投标行为，投标人所投标书作为废标处理：不同投标人的投标文件错、漏之处一致的；总报价相近，但其中分项报价不合理，没有合理的解释或者提不出计算依据或者主要材料设备价格极其相近或者没有成本分析，乱调乱压的；不同投标人的投标综合单价或者报价组成异常一致或者呈规律性变化的。其他国家法律、法规及有关规定的情形。六、定标1本次招标人授权评标委员会确定中标人。2定标办法2.1本次招标采用综合评分法。综合评分法是指在最大限度地满足招标文件实质性要求前提下，评标委员会按照招标文件中规定的各项因素进行综合评审和比较后，以评标总得分最高的投标人作为中标人（评分标准详见附表）。2.2评标委员会根据招标文件对投标文件的响应情况进行评分，合格投标人技术得分为评标委员会评分的算术平均值。按评审后报价、商务、技术之和得分排名由高到低对投标人进行排位，确定排位第一的投标人中标。若综合得分相同，投标报价低的投标人中标，报价仍相同的，由评标委员会确定技术部分最优的投标人中标。若排名第一的投标人放弃中标或不能按规定履行合同，由第二名递补，依次顺延。3对未中标的投标人，招标人有权不作任何解释。七、关于投标人瑕疵滞后发现的处理规则1无论基于何种原因，本应作无效、废标处理的情形即便未被及时发现而使该投标人进入初审、详细评审或者其它后续程序，包括已经签约的情形，一旦在任何时间被发现存在上述情形，评标委员会均有权随时视情形决定是否取消该投标人的此前评议结果，或者随时视情形决定该投标无效，并有权决定采取相应的补救、纠正措施；若通过补救、纠正措施能够满足招标文件或者招标人要求，评标委员会可以维持既定结果并要求中标人出具补救、纠正措施等承诺；若通过补救、纠正措施仍不能够满足招标文件或者招标人要求，评标委员会应出具取消该投标人的此前评议结果的复审结论，并予以废标，由此产生的一切损失均由中标人承担。评标委员会认定中标人投标无效、废标或者中标人的此前评议结果被取消的，根据评标结果，由第二名递补，依此类推。2若已经超过质疑期限而没有被发现，签署了相关的合同之后才发现存在上述情形，经评标委员会再行审查认为其在技术、必要资质等方面并不存在问题而仅属于商务方面存在瑕疵的问题，若取消该投标人的此前评议结果或者采取类似的处理措施将对本次招标采购更为不利的情形（包括：予以无效投标、废标或者采取类似的处理措施将使本次招标采购成本大幅上升、延误期限以至可能给招标人造成较大损失的），维持中标结果的，招标人必须出具维持中标结果以及是否要求提供特别担保金的书面意见，评标委员会可以维持既定结果并要求中标人出具提供特别担保金承诺，以承担可能产生的赔偿责任；若中标人拒绝提供特别担保金、实际提供的担保金额不足或者招标人不同意维持中标结果的，评标委员会应当决定取消中标人的此前评议结果或者采取类似的处理措施，由此产生的一切损失均由中标人承担。八、投标无效出现下列情形之一的，投标无效：1投标报价超出招标控制价或者投标文件未按照规定制作、盖章的；2对招标文件中实质性内容未做出实质性响应或者发生重大偏离的；3不按照规定报价、拒绝报价或者报价超过招标控制价的；4投标人法定代表人或者委托代理人未按时参加开标会议或者参加开标会议未提供有效证明，以及投标人复制招标文件的技术规格相关部分内容作为其投标文件的一部分的；5无投标人法定代表人或者其授权代表签字的；6投标有效期不满足招标文件要求或者有多个投标报价的；7超出经营范围投标的；8评标委员会2/3以及以上成员认定投标方案技术含量低、不符合招标文件要求或者无效报价的；10评标委员会判定投标人涂改证明材料或者提供虚假材料的；11本招标文件规定的投标无效情形的；12不符合法律、法规和招标文件中规定的其他要求的。对投标无效的认定，必须经评标委员会集体做出决定并出具投标无效的事实依据，由投标人法定代表人或者被授权代表签字确认，拒绝签字的，不影响评标委员会做出的决定。九、废标出现下列情形之一的，应予废标：1符合条件的投标人或者对招标文件作实质响应的投标人不足三家的；2出现影响采购公正的违法、违规行为的；3投标人的报价均超过了招标控制价的；4因重大变故，采购任务取消的；5法律、法规以及招标文件规定废标情形。十、在任何评标环节中，需评标委员会就某项定性的评审结论做出表决的，由评标委员会全体成员按照少数服从多数的原则，以记名投票方式表决。十一、违法违规情形1有下列情形之一的，属于投标投标人相互串通投标：1.1投标人之间协商投标报价等投标文件的实质性内容；1.2投标人之间约定中标人；1.3投标人之间约定部分投标人放弃投标或者中标；1.4属于同一集团、协会、商会等组织成员的投标人按照该组织要求协同投标；1.5投标人之间为谋取中标或者排斥特定投标投标人而采取的其他联合行动。2有下列情形之一的，视为投标人相互串通投标：2.1不同投标人的投标文件由同一单位或者个人编制；2.2不同投标人委托同一单位或者个人办理投标事宜；2.3不同投标人的投标文件载明的项目管理成员为同一人；2.4不同投标人的投标文件异常一致或者投标报价呈规律性差异；2.5不同投标人的投标文件相互混装；2.6不同投标人的投标保证金从同一单位或者个人的账户转出。3有下列情形之一的，属于招标人与投标投标人串通投标：3.1招标人在开标前开启投标文件并将有关信息泄露给其他投标人;3.2招标人直接或者间接向投标人泄露标底、评标委员会成员等信息；3.3招标人明示或者暗示投标人压低或者抬高投标报价；3.4招标人授意投标人撤换、修改投标文件；3.5招标人明示或者暗示投标人为特定投标人中标提供方便；3.6招标人与投标人为谋求特定投标人中标而采取的其他串通行为。4投标人有下列情形之一的，属于投标人弄虚作假的行为：4.1使用伪造、变造的许可证件；4.2提供虚假的财务状况或者业绩；4.3提供虚假的项目负责人或者主要技术人员简历、劳动关系证明；4.4提供虚假的信用状况；4.5其他弄虚作假的行为。评标过程中第60条规定情形之一的，评标委员会必须出具违法违规认定意见并予以废标。十二、违规处理投标人有下列情形之一的，列入不良行为记录名单，在一至三年内禁止参加枣庄烟草有限公司招标投标活动：1提供虚假投标材料谋取中标、成交的；2采取不正当手段诋毁、排挤其他投标人的；3与招标人、其他投标人或者招标代理机构恶意串通的；4向招标人、招标代理机构行贿或者提供其他不正当利益的；5在招标采购过程中与招标人进行协商谈判的；6拒绝有关部门监督检查或者提供虚假情况的。7一年内累计三次以上投诉均查无实据，并带有明显故意行为的；8捏造事实或者提供虚假投诉材料的；9不按照规定程序以及正常途径质疑、投诉，采用匿名信、匿名电话、发短信息等手段，威胁、恫吓、辱骂、恶意中伤其他相关当事人的；10法律、法规和招标文件中规定的其他情形。十三：付款方式：项目竣工验收合格后无质量问题，且经审计部门决算后，施工方提供增值税专用发票后付审定工程造价款的90％，剩余的10%作为质保金，待无质量问题二年后无息全部付清。审计费用按国家有关规定执行。施工单位结算申报金额与审定结果对比，当审减金额超过地方规定最低比例时，超过部分审计费用由施工单位承担。第八章纪律和监督一、对招标人的纪律要求招标人不得泄漏招标投标活动中应当保密的情况和资料，不得与投标人串通损害国家利益、社会公共利益或者他人合法权益。二、对投标人的纪律要求投标人不得互相串通或者与招标人串通投标，不得向招标人或者评标委员会成员行贿谋取中标；不得以他人名义投标或者以其他方式弄虚作假骗取中标；投标人不得以任何方式干扰、影响评标工作。三、对评标委员会成员的纪律要求评标委员会成员不得收受他人的财物或者其他好处，不得向他人透漏对投标文件的评审和比较、中标候选人的推荐情况以及评标有关的其他情况。在评标活动中，评标委员会成员应当客观、公正地履行职责，遵守职业道德，不得擅离职守，影响评标程序正常进行，不得使用超出本招标文件有关规定的评审因素和评标标准进行评标。68、对与评标活动有关的工作人员的纪律要求与评标活动有关的工作人员不得向他人透漏对投标文件的评审和比较、中标候选人的推荐情况以及评标有关的其他情况。在评标活动中，与评标活动有关的工作人员不得擅离职守，影响评标程序正常进行。第九章质疑与投诉一、质疑1按照有关规定，参加本次采购活动的投标人认为招标文件、招标过程和中标结果使自己的权益受到损害的，可以在知道或者应知道其权益受到损害之日起五个工作日内，以书面形式向招标代理机构提出质疑。2开标异议：投标人对开标有异议的，应当在开标现场举手提出，会后不再受理。招标人当场作出答复，并制作记录。第十章工程量清单及图纸级索服务站：龙阳服务站：羊庄服务站：第十一章技术标准和要求一、本工程项目的材料、设备、施工必须达到现行中华人民共和国及省、市、行业的一切有关法规、规范的要求，如各标准及规范要求有出入则以较严格者为准。二、执行国家、省、市现行的建设工程施工及工程质量验收规范、施工技术标准、程序，建设工程施工操作规程、“建设工程质量管理条例”、“工程建设强制性标准”、“工程建设标准强制性条文”、“建筑工程安全生产管理条理”、“建筑施工安全检查标准”以及有关建筑质量、安全施工、建筑材料及半成品备案证制度等有关文件、规定、施工图纸、技术交底等有关技术说明。第十二章评标办法一、相关要求1当投标人未提供符合招标文件规定的技术支持资料时，其技术部分得0分。2技术汇总得分的计算方法：所有评标委员会成员得分的算术平均值。3“同类项目”是指投标人已经完成的与本次招标性质相似的工程，并且签订合同一方必须是投标人，其下属具有独立法人资格的公司所有业绩不计，以相同或者类同部分的合同金额为准。4投标人总得分为报价分、商务得分、技术得分之和。附：评分标准注：1、同一工程业绩只计取最高级别得分，不累计记分；2、所有加分项开标时必须提供原件，否则不加分。业绩必须同时提供施工合同原件；3、本招标文件所要求近三年均指至今；近两年度指至今。第十三章合同发包人（全称）：山东枣庄烟草有限公司承包人（全称）：根据《中华人民共和国合同法》、《中华人民共和国建筑法》及有关法律规定，遵循平等、自愿、公平和诚实信用的原则，双方就及有关事项协商一致，共同达成如下协议：一、项目概况1.项目名称：。2.项目实施地点：。3.项目立项批准文号：。4.资金来源：。5.项目内容：6.项目承包范围：二、合同工期计划开工日期：具体的开工日期，以工程施工现场具备开工条件后，发包人和监理单位共同下达的开工令中日期为准。如发包人和监理单位认为具备开工条件，但承包人不呈报开工报告，在发包人和监理单位书面通知其呈报后的三天内，承包人仍不报送开工报告的，发包人和监理单位有权直接下达开工令，且以该开工令作为计算工期的开始之日。计划竣工日期：年月日。工期总日历天数：35日历天。工期总日历天数与根据前述计划开竣工日期计算的工期天数不一致的，以工期总日历天数为准。三、质量标准工程质量达到合格。四、签约合同价与合同价格形式1、合同价款：元整（大写）圆整。2.付款形式：项目竣工验收合格后无质量问题，且经审计部门决算后，施工方提供增值税专用发票后付审定工程造价款的90％，剩余的10%作为质保金，待无质量问题二年后无息全部付清。审计费用按国家有关规定执行。施工单位结算申报金额与审定结果对比，当审减金额超过地方规定最低比例时，超过部分审计费用由施工单位承担。五、项目经理承包人项目经理：。六、合同文件构成本协议书与下列文件一起构成合同文件：（1）中标通知书；（2）投标函及其附录；（3）专用合同条款及其附件；（4）通用合同条款；（5）技术标准和要求；（6）图纸；（7）已标价工程量清单或预算书；（8）招标文件；（9）其他合同文件。在合同订立及履行过程中形成的与合同有关的文件均构成合同文件及以上合同文件为本合同的组成部分，与本合同有同等法律效力。上述各项合同文件包括合同当事人就该项合同文件所作出的补充和修改，属于同一类内容的文件，应以最新签署的为准。专用合同条款及其附件须经合同当事人签字或盖章。七、承诺1.发包人承诺按照法律规定履行项目审批手续、筹集工程建设资金并按照合同约定的期限和方式支付合同价款。2.承包人承诺按照法律规定及合同约定组织完成工程施工，确保工程质量和安全，不进行转包及违法分包，并在缺陷责任期及保修期内承担相应的工程维修责任。3.发包人和承包人通过招投标形式签订合同的，双方理解并承诺不再就同一工程另行签订与合同实质性内容相背离的协议。4.本工程不得分包或转包。八、词语含义本协议书中词语含义与第二部分通用合同条款中赋予的含义相同。九、签订时间本合同于年月日签订。十、签订地点本合同在签订。十一、补充协议合同未尽事宜，合同当事人另行签订补充协议，补充协议是合同的组成部分。十二、合同生效本合同自双方共同签字、盖章后生效。十三、合同份数本合同一式陆份，均具有同等法律效力，发包人执叁份，承包人执叁份。十四、发包人与承包人发生争议，双方应协商解决；如协商不成，可依法向发包人所在地有管辖权的人民法院起诉。发包人：(公章)承包人：(公章)法定代表人或其委托代理人：法定代表人或其委托代理人：（签字）（签字）组织机构代码：组织机构代码：地址：滕州市学院路599号地址：邮政编码：邮政编码：法定代表人：法定代表人：委托代理人：委托代理人：电话：0632-5501607电话：传真：传真：电子信箱：电子信箱：开户银行：农行滕州市支行营业部开户银行：户名：山东枣庄烟草有限公司户名：账号：260101040007529账号：第十四章投标文件格式报价一览表单位：人民币元投标人名称：（盖章）法定代表人或授权代表：（签名或盖章）日期：年月日（一）投标函（招标人名称）：1．我方已仔细研究了（项目名称）工程之招标文件的全部内容，愿意以人民币（大写）（¥）的投标总报价，工期日历天，按照招标文件、施工合同、设计文件、工程量清单、技术规范等承接上述工程的施工、竣工和保修等任务。2．我方承诺在招标文件规定的投标有效期内不修改、撤销投标文件。3．随同本投标函提交投标保证金一份，金额为人民币（大写）（¥）。4．如我方中标：（1）我方承诺在收到中标通知书后，在中标通知书规定的期限内与你方签订合同。（2）随同本投标函递交的投标函附录属于合同文件的组成部分。（3）我方承诺按照招标文件规定向你方递交履约担保。（4）我方承诺在合同约定的期限内完成并移交全部合同工程。5．我方在此声明，所递交的投标文件及有关资料内容完整、真实和准确，且不存在违反招标文件规定的任何一种情形。投标人：（公章）法定代表人或授权代理人：（签字）年月日（二）法定代表人身份证明投标人名称：单位性质：地址：成立时间：年月日经营期限：姓名：性别：年龄：职务：系（投标人名称）的法定代表人。特此证明。附：法定代表人身份证复印件。投标人：（公章）年月日（三）法定代表人授权委托书本人（姓名）系（投标人名称）的法定代表人，现委托（姓名）为我方代理人。代理人根据授权，以我方名义签署、澄清、说明、补正、递交、撤回、修改（项目名称）设计施工总承包投标文件、签订合同和处理有关事宜，其法律后果由我方承担。代理人无转委托权。附：法定代表人身份证明投标人：（盖单位章）法定代表人：（签字）身份证号码：被授权代表（委托代理人）：（签字），年龄：岁，职务：身份证号码：附身份证复印件授权委托日期：20年月日法定代表人电话：通讯地址：（四）本工程配备的工程管理成员一览表注：本表须后附项目班子成员的资格证书复印件并提交原件，否则该分项不得分。投标人：（公章）法定代表人或投标人全权代表：（签字或印章）日期：年月日（五）投标文件密封信封正面格式（六）投标文件密封信封封口格式山东枣庄烟草有限公司标准服务站维修项目技术标书"]
    for text in texts:
        res_ = ner_softmax_predict.predict(text)
        print(res_)
        print('*' * 80)

        print(text)
        # print(res)
        for res in res_:
            for x in res['entities']:
                start = int(x[1])
                end = int(x[2]) + 1
                ner_obj = text[start:end]
                print(x[0])
                print(ner_obj)