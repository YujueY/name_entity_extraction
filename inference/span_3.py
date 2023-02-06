# -*- coding: utf-8 -*- 
import copy
import json
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset, DistributedSampler

from BERT_NER_Pytorch.processors.ner_span import InputFeature
from model_configs.contract_argparse import get_argparse
from BERT_NER_Pytorch.models.transformers import BertConfig, AlbertConfig
from BERT_NER_Pytorch.models.bert_for_ner import BertSpanForNer
from BERT_NER_Pytorch.models.albert_for_ner import AlbertSpanForNer
from BERT_NER_Pytorch.processors.utils_ner import CNerTokenizer
from BERT_NER_Pytorch.tools.common import logger
from utils.process_data import split_sentences
from utils.utils_ner import bert_extract_item_v2


class GeneralModelSpanPredict():
    def __init__(self, checkpoint, max_seq_length):
        self.checkpoint = checkpoint
        self.args = get_argparse().parse_args()
        self.max_seq_length = max_seq_length

        MODEL_CLASSES = {
            ## bert ernie bert_wwm bert_wwwm_ext
            'bert': (BertConfig, BertSpanForNer, CNerTokenizer),
            'albert': (AlbertConfig, AlbertSpanForNer, CNerTokenizer),
            'roberta':(BertConfig, BertSpanForNer, CNerTokenizer),
        }

        # self.label_list = ["O", "合同变更的条件", "合同生效条件", "合同终止的条件", "支付方式", "法律依据", "解决方式"]
        self.label_list = ["O", '交货时间', '交货地点', '定金条款', '履约保证金支付工具', '履约保证金金额', '履约保证金支付时间', '履约保证金返还约定', '质保金支付工具', '质保金支付时间', '质保金返还约定', '银行信息变更', '发票条款', '发票税率', '发票信息变更约定', '质保期', '质保期开始时间', '知识产权无瑕疵保证', '甲方保密期限', '乙方保密期限', '迟延交货违约责任', '质量瑕疵违约责任', '其他违约责任', '违约金上限', '甲方免责条款', '乙方免责条款', '甲方间接损失', '乙方间接损失', '甲方赔偿限额', '乙方赔偿限额', '不可抗力免责', '不可抗力情形', '不可抗力通知义务', '不可抗力通知期限', '不可抗力证明', '不可抗力证明机构', '不可抗力证明期限', '送达接收方通知', '甲方合同解释权', '乙方合同解释权']
        self.id2label = {i: label for i, label in enumerate(self.label_list)}
        self.label2id = {label: i for i, label in enumerate(self.label_list)}

        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.args.model_type]
        config = config_class.from_pretrained(self.checkpoint,
                                              num_labels=len(self.label_list),
                                              cache_dir=None)

        self.tokenizer = tokenizer_class.from_pretrained(self.checkpoint,
                                                         do_lower_case=self.args.do_lower_case,
                                                         cache_dir=None)

        self.model = model_class.from_pretrained(self.checkpoint, config=config)

        self.model.to(self.args.device)

    def predict(self, texts):
        if not texts:
            return {}
        text_list = split_sentences(texts, 510)
        # texts = texts.split('。')
        # text_list = [i + '。' for i in texts if i]

        test_dataset = self.load_and_cache_examples(text_list, self.args, self.tokenizer)
        # print(len(test_dataset))
        # Note that DistributedSampler samples randomly
        test_sampler = SequentialSampler(test_dataset) if self.args.local_rank == -1 else DistributedSampler(
            test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=self.args.batch_size,
                                     collate_fn=collate_fn)  # self.args.per_gpu_eval_batch_size

        results = defaultdict(list)
        pre_length = 0
        new_text_list = []
        idx_txt = 0
        while idx_txt < len(text_list):
            new_text_list.append(text_list[idx_txt: idx_txt + self.args.batch_size])
            idx_txt += self.args.batch_size
        for txt_batch, (step, batch) in zip(new_text_list, enumerate(test_dataloader)):  # 1句话一个batch
            self.model.eval()
            batch = tuple(t.to(self.args.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "start_positions": None,
                          "end_positions": None}
                if self.args.model_type != "distilbert":
                    # XLM and RoBERTa don"t use segment_ids
                    inputs["token_type_ids"] = (batch[2] if self.args.model_type in ["bert", "xlnet"] else None)
                outputs = self.model(**inputs)
            start_logits_list, end_logits_list = outputs[:2]
            for txt, start_logits, end_logits in zip(txt_batch, start_logits_list, end_logits_list):
                R = bert_extract_item_v2(start_logits[:len(txt)+2], end_logits[:len(txt)+2])
                #R = bert_extract_item_v2(start_logits, end_logits)
                # txt = txt.replace("\x00", "")
                txt_length = len(txt)
                for x in R:
                    results[self.id2label[x[0]]].append([txt[x[1]:x[2] + 1], pre_length + x[1], pre_length + x[2] + 1])
                pre_length += txt_length
        print("batch_size", "*"*10, self.args.batch_size)

        return results

    def load_and_cache_examples(self, texts, args, tokenizer):
        class InputExample(object):
            """A single training/test example for token classification."""

            def __init__(self, guid, text_a):
                self.guid = guid
                self.text_a = text_a

            def __repr__(self):
                return str(self.to_json_string())

            def to_dict(self):
                """Serializes this instance to a Python dictionary."""
                output = copy.deepcopy(self.__dict__)
                return output

            def to_json_string(self):
                """Serializes this instance to a JSON string."""
                return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

        logger.info("Creating features")
        examples = []
        for text in texts:
            text = list(text)
            # labels = ["O"] * len(text_)
            # subject = get_entities(labels, id2label=None, markup='bios')
            examples.append(InputExample(guid="001", text_a=text))

        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                label_list=self.label_list,
                                                max_seq_length=self.max_seq_length,
                                                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in [
                                                    "xlnet"] else 0,
                                                sep_token=tokenizer.sep_token,
                                                pad_token=tokenizer.convert_tokens_to_ids(
                                                    [tokenizer.pad_token])[
                                                    0],
                                                pad_token_segment_id=4 if args.model_type in [
                                                    'xlnet'] else 0
                                                )

        max_length = max([f.input_len for f in features])

        all_texts = [[ord(char) for char in text] for text in texts]
        padding_all_texts = []
        for i in all_texts:
            assert len(i) <= max_length
            if len(i) < max_length:
                len_i = len(i)
                i.extend([0] * (max_length - len_i))
                padding_all_texts.append(i)
            else:
                padding_all_texts.append(i)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_start_ids = torch.tensor([f.start_ids for f in features], dtype=torch.long)
        all_end_ids = torch.tensor([f.end_ids for f in features], dtype=torch.long)
        all_input_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_ids, all_end_ids,
                                all_input_lens)
        return dataset


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_input_mask, all_segment_ids, all_start_ids, all_end_ids, all_lens = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_input_mask = all_input_mask[:, :max_len]
    all_segment_ids = all_segment_ids[:, :max_len]
    all_start_ids = all_start_ids[:, :max_len]
    all_end_ids = all_end_ids[:, :max_len]
    return all_input_ids, all_input_mask, all_segment_ids, all_start_ids, all_end_ids, all_lens


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer,
                                 cls_token_at_end=False, cls_token="[CLS]", cls_token_segment_id=1,
                                 sep_token="[SEP]", pad_on_left=False, pad_token=0, pad_token_segment_id=0,
                                 sequence_a_segment_id=0, mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # label2id = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        textlist = example.text_a
        # subjects = example.subject
        tokens = tokenizer.tokenize(textlist)
        start_ids = [0] * len(tokens)
        end_ids = [0] * len(tokens)
        # subjects_id = []
        # for subject in subjects:
        #     label = subject[0]
        #     start = subject[1]
        #     end = subject[2]
        #     start_ids[start] = label2id[label]
        #     end_ids[end] = label2id[label]
        #     subjects_id.append((label2id[label], start, end))
        # Account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            start_ids = start_ids[: (max_seq_length - special_tokens_count)]
            end_ids = end_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        tokens += [sep_token]
        start_ids += [0]
        end_ids += [0]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        if cls_token_at_end:
            tokens += [cls_token]
            start_ids += [0]
            end_ids += [0]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            start_ids = [0] + start_ids
            end_ids = [0] + end_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            start_ids = ([0] * padding_length) + start_ids
            end_ids = ([0] * padding_length) + end_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            start_ids += ([0] * padding_length)
            end_ids += ([0] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(start_ids) == max_seq_length
        assert len(end_ids) == max_seq_length

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s", example.guid)
        #     logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
        #     logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
        #     logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
        #     logger.info("start_ids: %s" % " ".join([str(x) for x in start_ids]))
        #     logger.info("end_ids: %s" % " ".join([str(x) for x in end_ids]))

        features.append(InputFeature(input_ids=input_ids,
                                     input_mask=input_mask,
                                     segment_ids=segment_ids,
                                     start_ids=start_ids,
                                     end_ids=end_ids,
                                     input_len=input_len))
    return features


if __name__ == "__main__":
    max_seq_length = 512
    checkpoint = "./models/chinese_roberta_L-4_H-512_span_总体+_v5"
    model = GeneralModelSpanPredict(max_seq_length=max_seq_length, checkpoint=checkpoint)


    text = "绿化养护合同发包人：（以下称甲方）大兴区园林服务中心南区公园管理所法定代表人／负责人：梁艳联系地址：北京市大兴区兴旺路兴旺公园内联系电话：60217735承包人：（以下称乙方）北京海景国宏保法服务有限公司法定代表人／负责人：联系地址：联系电话：1391695800甲方委托乙方承担黄村公园、街心公园绿化养护项目养护管理工作。根据《中华人民共和国民法典》及其他相关法律、法规的规定，甲乙双方遵循平等、自愿、公平和诚实信用的原则，双方经协商订立本合同，共同执行。第一条本合同签订依据1.1《中华人民共和国民法典》、DB11／T213-2014《城镇绿地养护管理规范》、中标通知书。1.2国家及地方有关法规和规章。第二条养护项目基本情况2.1养护项目名称：南区公园管理所绿化养护项目（黄村公园、街心公园）。2.2养护地点及养护面积：黄村公园、街心公园；养护面积5.86万平方米。2.3资金来源：政府资金2.4甲方负责养护管理项目联系人：张亮2.5乙方负责养护管理项目代表人：姜沛琰第三条养护期限3.1合同履行期（服务期限）：自2022年1月1日起至2022年6月30日止。第四条养护标准4.1乙方负责按照北京市《城市园林绿化养护管理标准》、《城市园林绿化养护管理规范》（DB11／T213-2014）及《大兴区园林服务中心管理考核工作方案》履行养护工作，严格履行本合同约定的相关义务。4.2本公园按照《城市园林绿化养护管理标准》的特级标准进行养护。4.3甲方按第4.1条约定对乙方进行考核。第五条本合同文件及解释顺序5.1除双方另有约定以外，组成本合同的文件及优先解释顺序如下：（1）双方签订的补充协议（2）本合同条款（3）中标通知书（4）投标书及其附件（5）招标文件（6）标准、规范及有关文件（7）图纸（8）工程量确认单（9）工程报价单或预算书（10）DB11／T213-2014《城镇绿地养护管理规范》（11）绿化养护服务项目廉政责任书5.2双方在本合同履行中所共同签署或认可的符合现行法律、法规、规章及规范性文件，以及政府有关部门及甲方主管部门就本合同约定项目的实质性约定的指令、纪要或有关文件等，均构成合同文件的有效补充，甲乙双方均应执行。第六条乙方养护与卫生保洁管理工作内容包括但不仅限于以下养护管理工作：绿化养护工作内容：6.1公园绿地养护工作包含但不限于：公园绿地园林绿化养护日常管理工作，如除草、浇水、施肥、清理绿地垃圾、搂树叶、打药、日常枯枝死树、残花的清理、草坪、色带、花灌木、乔木、行道树的修剪整形，垃圾的清运和处置，养护期内出现的针对苗木死亡的补栽补种工作等，园区如存在古树需按照相关标准进行复壮和养护；公园内的水面保持清洁，对于极端天气（如大风、大雪、汛期等）要做好应急抢险工作，园区内环境的清理工作。对公园内广场、硬化路面、公厕、景观小品等附属设施的清洁及消毒消杀工作，公厕或其他的基础设施等消毒清洁物资消耗品的供给，按照甲方要求做好疫情防控、创城创卫工作。6.2园林机械及所用汽柴油和绿地内产生的人工费（如更换地上灌溉设施维修维护等）等所有消耗及费用均由乙方承担。6.3综合要求6.3.1着装：所有现场作业人员必须按要求进行统一着装并佩带胸卡。（具体样式需得到甲方确认同意，每个养护人员至少配备4套，春夏各2套），服装每天保持整洁，同时配备雨衣等雨天作业服装。6.3.2养护作业时间：作业时间安排根据气候及甲方养护要求进行，在合同签订后5个工作日内，乙方应向甲方提交本年度绿化养护工作计划，详细到完成具体道路具体工作的起止时间，投入人员，机械、车辆，必须包含修剪、补植、打药、浇水、除草等详细工作。以后于每月25日前提交下月的工作方案，工作方案经甲方审核确认后实施。6.3.3机械：必须根据工作计划并结合实际情况配备相应浇水车、升降工作车、农药喷洒机、剪草机、疏草机、修边机、打孔机、绿篱机等等。不得使用简易洒水设备（车辆）。6.3.4修剪：修剪季相明显，树冠完美，分枝点合适，无枯死枝；主侧枝分布均匀、数量适宜，通风透光。截口平整，不拉伤树皮，截口涂抹防腐剂。具体修剪次数根据养护技术规范的要求执行，要求乔木每年整形修剪1次，于12月30日前完成，植物生长旺盛季节，发现脚芽须及时清除（芽长不得超过10公分），剥牙时不允许拉伤树皮。花灌木要求每年整形修剪1次，部分花灌木需进行两次修剪即花后修剪；绿篱要求一次修剪成型，以后按新长出的枝高度进行修剪（枝叶高度不超过10公分修剪一次），全年修剪次数不少于6次；草坪修剪次数：冷季型不少于10次，暖季型为3次。修剪高度控制在冷季型草坪为10公分，暖季型草坪为6公分以下。草坪边角必须切边。月季、萱草等残花须随时出现随时修剪。6.3.5浇水：植物浇水应根据不同季节、植株生长不同阶段及时进行，且一次浇透。春季干旱，要求每周浇水不少于2次；夏季结合降雨量调解浇水量，雨涝部分要及时排水，高温季节严禁中午浇水；其它季节每周浇水不少于1次，入冬前必须浇一次灌冬水，要求浇足浇透；温度低于0℃，不能进行浇水作业。新栽植株每天浇水2次。草坪：必须浇透根系层，冷季型草坪在春秋两季充分浇水，暖季型草坪夏季要勤浇水。浇水作业过程中避免对路面造成污染，如有污染必须及时整改，由此造成的后果，养护作业单位应负完全责任。6.3.6施肥：每年春秋两季施腐熟有机复合肥不少于500克／平方、氮磷钾复合肥不少于45克／平方。主要干道的行道树每年施一次腐熟菜饼肥，用量为1000克／株。6.3.7病虫害防治：根据植株生长特性及各生长阶段容易发生的病虫害，及时做好病虫害的预测，预防工作。预防规模型病虫害发生。发生病虫害时及时防治，用药配比正确，不允许发生因病虫害而导致的植株死亡现象。喷药时间宜在清晨或傍晚，喷药前必须提前联系居委会住区物业等属地部门，张贴喷药告示至临街小区各个楼门，以此通知居民，关好门窗，不得影响市民正常行为。草坪修剪后1-2天，应喷洒一次常规抗菌剂（如甲基托布津），6-8月病害多发季节，应加大喷洒密度。乔灌木入冬前及时涂白，涂白时间为每年十一月中旬至十二月上旬完成，涂白所用材料，浓度配比需按照北京市相关要求执行。不得使用广谱性高毒杀虫农药（敌敌畏、敌百虫、辛硫磷等），严禁使用国家禁止使用的农药（甲氨磷、呋喃丹等）。6.3.8松土：植株根部周围土壤要保持疏松，春、夏、冬季必须松土1次。易板结的土壤在蒸腾旺季每月松土不少于1次。松土深度以不伤根系生长为限：乔木30公分以上，花灌木15公分以上，草坪10公分以上。松土范围：乔木为树穴直径1.5米范围以上，花灌木为植株冠幅。6.3.9除草：在植株生长旺盛期须及时地进行除草，不允许有规模型的杂草生长、爆发，树穴内无杂草。其它时期进行定期除草。6.3.10树木支撑：新栽、倾斜树木必须进行扶正支撑；乔木、竹类植株等必须有防风措施。支撑所用材料须做到合理、美观，不伤及到树木本身，支撑所用材料应在规定时间内拆除。6.3.11补植：所有植株保存率须达到100％。发现枯枝、死枝、踩踏须24小时内处理完毕；发现有枯死树木须在两星期内补植完毕，并确保其成活；花灌木的补植须在三天内完成；草坪空秃补植须在一周内完成。行道树发生死亡，2天内更换。所有补植补造的苗木必须与原苗木品种、数量规格相一致，不得以小充大，以次充好，以少充多。特殊情况须报甲方主管部门，经其同意后，可做适当调整。6.3.12防寒方融雪剂：6.3.12.1绿地防寒：四面围挡；围挡上沿要高于植物材料5-10厘米；统一使用草绿色防寒布；龙骨全部采用木质条方。防寒围挡面层绑扎平整牢固，做到横平竖直、见棱见角，无松垮、漏风现象。同一道路绿化防寒，围挡高度保持一致。6.3.12.2防融雪剂：围挡上沿要高于植物材料5-10厘米；统一使用草绿色铁质或玻璃钢制挡盐板。6.3.12.3丛生花灌木和常绿乔木的防寒：沿迎风面两侧或三侧搭设正方型围挡；统一使用草绿色防寒布；龙骨主架采用标准脚手架钢管及连接件。防寒围挡表层绑扎平整牢固，做横平竖直、见棱见角，无松垮、漏风现象。6.3.13损坏赔偿：乙方应加强白天及晚上的巡视，及时发现和阻止各种违法施工损坏行为及人为损坏，并及时上报；由于乙方未能及时发现和阻止的，由乙方自行负责承担，其风险费用应全部包含在本合同养护费报价中。6.3.14绿地内附属设施要每周进行巡查，以不影响市民使用为原则，及时维修理，避免出现举报投诉现象；如出现上述情况，由乙方承担由此造成的全部损失。6.3.15乙方所安排人员和机具设备仅限于本项目所用，不得套用。6.3.16乙方的用工（包括管理人员和养护作业人员等）必须符合国家用工的相关规定。6.3.17应及响应及时，在接到应急任务时15分钟内工作人员到达指定场所，到达人数必须符合甲方应急人员要求人数。6.3.18乙方对园区内的广场、道路、廊亭、桥梁、台阶、井盖、电箱、栏杆、垃圾桶、座椅等一切的构筑物、建筑物、景观小品等进行全天候的保洁清理，确保无污渍、无生锈、无尘积、无杂物、无痰迹、无烟蒂、墙面或外立面无脱落。6.3.19古树复壮及养护按照区级园林绿化行业主管部门和甲方的要求进行养护。6.3.20厕所要每日进行清理清洁，做到随脏随保，达到无异味、无污渍，便池、蹲坑无积粪、无尿垢，玻璃、扶手保持清洁，厕位保持水冲式标准功能性，洗手台前要供给防疫及消杀的物品，做好消杀和清理的工作台账。6.3.21园区要进行生活垃圾分类管理，实行袋装化、桶装化、有专人定时收集，垃圾清运密闭化。果皮箱每日进行清理，消杀，严禁出现果皮箱满冒、脏污现象，所有生活垃圾做到日产日清。落实“除四害”工作。6.3.22做好园区换水及清淤工作，保持园内水面的清洁，每天定时打捞漂浮物、枯枝落叶、垃圾等，做好夏冬两季的水生植物的修剪和垃圾清理工作。第七条双方权利和义务7.1甲方权利和义务7.1.1甲方有权全面监督、指导、检查、考核、验收乙方工作。根据《城市园林绿化养护管理标准》审定乙方的年度管护方案，对合同期内乙方的现场养护工作进行全面的指导、监督和验收，检查和督促乙方做好内业资料，审定乙方的养护经费或进度款项的拨付情况。7.1.2因乙方安全措施未落实或绿化养护质量未达到合同约定标准，甲方有权发出停工整改令，乙方应按甲方的停工整改令执行，并不得因停工整改而提出任何费用增加的要求。7.1.3甲方检查时如发现有不合格之处，有权以“考核整改通知书”的形式书面通知乙方，乙方应在通知要求时间内进行整改。7.1.4因乙方原因使养护项目达不到本合同约定标准及甲方考核标准的，甲方有权要求乙方在通知规定期限内内无偿修改或返工，并有权扣除或预留一定比例的养护费用（具体数额由甲方根据实际情况酌定），当月应付款项直到验收合格后再向乙方支付。7.1.5甲方管理人员有权当场制止乙方人员违反第4.1条约定及甲方管理制度、安全规范等的违章作业，并对乙方给予500-1000元经济处罚；7.1.6及时为乙方办理结算，按本合同约定支付应付费用。7.1.7甲方在合同期内因公园养护运营的需要可随时向乙方下达应急救援任务和其他临时性工作，包括疫情防控、创城创卫、安全维稳等工作。7.2乙方权利和义务7.2.1认真按照合同约定的养护管理标准及养护内容执行，遵守甲方的各项规章制度，服从甲方的管理。乙方应根据养护标准和园区的实际情况，制定公园养护的具体措施，报甲方审定。7.2.2定期向甲方申报季度养护方案，汇报养护管理情况及有关措施。7.2.3乙方应当对项目期内可能出现的不利于养护的各种自然和社会因素（包括但不限于大风、降雨、降雪、沙尘暴、国家庆典、外宾或领导来访、高考、“两会”、周边民扰或扰民）做出充分预见，并提前制定周密的应对工作方案，乙方不得因上述因素造成停工或效率降低，并不得因此而向甲方提出费用增加要求。7.2.4为保证管理到位，乙方应配备专职管理人员，做好日常养护记录，建立养护技术档案。7.2.5乙方必须重视安全生产工作，负责对施工及参与人员进行安全教育，确保合同履行期间不出安全责任事故。合同履行期间，乙方人员发生的人身损害或财产损失及给甲方或他人造成的人身损害及财产损失等，以及其它安全责任事故等；均由乙方承担一切责任，并及时处理。如乙方未及时处理，造成相关人员向甲方或有关部门上访、围堵大门、办公场地或者可能给甲方造成其它不利影响时，甲方可以（但无义务）进行处理或先行垫付有关费用；无论甲方垫付的费用是否合理均由乙方承担，甲方有权向乙方追偿或在应付乙方款项中扣除，并要求乙方按甲方垫付部分的30％承担违约责任，且甲方有权解除合同。7.2.6无条件完成甲方安排的应急抢险任务和临时性工作，包括疫情防控和创城创卫、安全维稳等工作。7.2.7根据养护管理内容及管护标准编制年度养护管理方案并报甲方书面审核。7.2.8编制管护人员岗前技术培训方案并报甲方书面审核。7.2.9编制年度补植补造计划并报甲方书面审核。7.2.10编制安全文明施工方案并报甲方书面审核。7.2.11编制针对暴风、暴雨、暴雪、冰雹、干旱、冻害等自然灾害应急预案以及抵抗风险的措施并报甲方书面审核。7.2.12对道路、围栏、机井、灌溉管线、围栏、排水沟渠等基础设施进行保洁维护。7.2.13养护期内因养护不当发现苗木死亡，乙方应按原设计品种、规格进行补植补造，相关费用已包含在合同价款中；若乙方不予补植的，则由甲方补植，相关费用从乙方当年养护费用中扣除，且乙方应按补植费用的30％向甲方支付违约金。7.2.14乙方不得擅自将本合同转包或将合同项目下无论是全部还是部分权利转让给第三方，如乙方将其合同转包或将合同权利转让的，该转包或权利转让无效，对甲方不产生法律效力，并视为乙方违约，甲方有权解除合同，并要求乙方向甲方支付相当于合同金额20％的违约金。7.2.15因乙方安全措施未落实或绿化养护质量未达到合同约定标准，甲方有权发出停工整改令，乙方不得因停工整改而提出任何费用增加的要求。7.2.16因乙方原因致使养护达不到甲方考核标准的，甲方有权要求乙方在2日内无偿修改或返工，并扣除一定比例的养护费用，直到验收合格后再拨付。7.2.17乙方应严格履行本合同第六条规定的养护与卫生保洁管理工作内容及第7.2条约定的义务。乙方履行义务不合格，达不到养护标准，甲方有权限期整改，整改发生的费用由乙方承担，甲方因此遭受的损失由乙方承担；乙方未按甲方要求及在规定时间内进行整改的，甲方有权解除合同。7.2.18当乙方不履行合同义务或不按合同约定履行义务时，给甲方造成损失的，按照实际损失额进行赔偿。损失数额无法计算时，按照合同金额的30％计算。7.2.19养护期内发生群众举报、12345案件、网格案件等，乙方需及时解决，并按甲方要求及时整改。7.2.20养护期内，乙方的管理人员、绿化养护工人及保洁工人必须经过礼仪培训，涉及到乙方在园区内的礼仪接待纳入到甲方的考核之中。7.2.21乙方要充分预估到因疫情原因造成工作人员来京或在京居住等情况，做好相应预案，不得因此影响绿地养护工作。7.2.22乙方负责为养护人员缴纳保险、支付报酬，乙方工作人员必须遵循安全生产规定，不得从事危险性高的作业，乙方工作人员在作业过程中发生的交通、工伤、溺水和死亡事故、劳动纠纷及其他安全事故等，均由乙方承担全部责任，并由乙方承担全部费用，甲方概不负责。7.2.23乙方负责处理在工作过程中与第三方发生的纠纷，并独立承担由此产生的法律责任。第八条合同价款与支付8.1合同价款本合同项下的绿化养护费每年每平米10元，6个月养护，总金额（大写）人民币：陆拾捌万叁仟零玖拾玖元伍角整：（小写）¥683099.5元每季度养护费金额：（大写）叁拾肆万壹仟伍佰肆拾玖元柒角伍分（小写）：¥341549.75元。本合同价款包含了乙方完成本合同义务和相关服务的全部费用，包括但不限于：人员、保险、利润、含税、材料费、设备、设施费、赔偿费等。除本合同另有明文规定外，甲方无需承担或向乙方支付任何其它费用或款项的义务。如需进行审计，待审计部门审计后，根据审计结果作为结算付款的依据。8.2养护费支付养护费用按甲方考核办法为依据，考通过后甲方每＿季度向乙方支付一次养护费。考核不合格的，乙方同意甲方酌情扣除一定比例的养护费用作为违约金，具体金额由甲方根据考核情况请甲方到现场检查。8.3付款条款甲方向乙方付款前，乙方应先向甲方提供合法、合规的税务发票，并经甲方验证通过并完成付款审批手续后付款。但甲方收取乙方发票，并不视为对乙方提供货物验收通过或对其提供货物及服务验收合格的确认。乙方向甲方提供发票的形式与内容均应合法、有效、完整、准确，乙方不开具或开具了不合格的发票，甲方有权迟延支付应付款项直至乙方开具合格票据之日，甲方不承担任何违约责任，乙方的各项合同义务仍应按合同约定履行。不合格发票包括但不限于以下情形：开具虚假、作废等无效发票或者违反国家法律法规开具、提供发票的；开具发票种类错误；开具发票税率不符合税法规定或与合同约定不符；发票上的信息错误；因乙方迟延送达、开具错误等原因造成发票认证失败等。甲方在收到乙方提供的正式有效发票之前，甲方有权暂停支付任何费用而无需承担逾期支付责任。因甲方系财政拨款单位，如因财政或有关部门就本项目资金未能及时拨款到位，待本项目资金到位后向乙方付款，而不视为甲方付款违约，甲方亦不承担任何违约责任。但乙方不得拒绝或延期履行义务，否则应按本协议约定承担违约责任。甲方收到乙方提供的发票后，无论任何时间发现乙方提供的发票不合格，乙方均应在甲方通知期限内予以重开并更换；如因此造成甲方被处罚或经济损失等由乙方承担，甲方有权向乙方追偿，并要求乙方按损失数额的30％向甲方承担违约责任。第九条不可抗力9.1双方关于不可抗力范围的约定：不可抗力除法律规定外。不可抗力一般包括以下的情况：（1）国家权威部门发布且被界定为灾害的瘟疫、地震、洪水、风灾、雪灾等；（2）战争；（3）离子辐射或放射性污染；（4）以音速或超音速飞行的飞机或其他飞行装置产生的压力波，飞行器坠落；（5）动乱、暴乱、骚乱或混乱，但完全局限在甲方及其乙方、聘用人员内部的事件除外；（6）因适用法律的变更或任何适用的后继法律的颁布所导致本合同的履行不再合法；9.2因不可抗力导致的费用及延误的工期由双方按以下方法分别承担：（1）甲方、乙方人员伤亡由其所在单位负责，并承担相应费用；（2）乙方机械设备损坏及停工损失，由乙方承担；（3）绿化养护所需清理、修复费用，由甲方承担；第十条合同解除10.1双方协商一致，可以解除合同。10.3对因管护不力导致森林失火、大面积病虫害、苗木死亡、损失重大或人员伤亡、重大财产损害的，甲方有权解除合同，要求乙方支付相当于合同金额30％的违约金，并要求乙方承担全部赔偿责任。10.4乙方将其承包的全部绿化养护转包给他人或者肢解以后以分包的名义分别转包给他人，甲方有权解除合同，并要求乙方支付相当于合同总金额30％的违约金。10.2甲方对乙方完成本合同项下养护项目连续两次验收不合格或连续两次考核不合格，甲方有权解除合同，并要求乙方支付相当于合同总金额30％的违约金。10.5乙方具有本合同约定的服务质量不合格等情形时，甲方可以解除合同，并要求乙方支付相当于合同总金额30％的违约金。10.6由于乙方管理原因（包括但限于：服务、管理、不适当履行合同义务等），造成管护区域内或周边居民上访，乙方不能及时妥善解决的，甲方有权解除合同，要求乙方支付相当于合同总金额30％的违约金。10.7有下列情形之一的，甲方、乙方可以解除合同：（1）因不可抗力致使合同无法履行；（2）因一方违约致使合同无法履行或合同目的无法实现的。10.8甲乙双方依据本合同约定行使合同解除权时，合同解除通知依据本合同约定方式送达之日起生效。有过错的一方应当承担违约责任，并赔偿因合同解除给对方造成的损失。合同按本合同约定程序或司法程序解除后，乙方应妥善做好已完养护保护和移交工作，按甲方要求将自有机械设备和人员撤出施工场地。甲方应为乙方撤出提供必要条件，所发生的费用由乙方承担，并按合同约定结算相关款项。除此之外，有过错的一方应当赔偿因合同解除给对方造成的损失。10.9合同解除后，不影响双方在合同中约定的结算和清理条款的效力。第十一条安全施工11.1乙方是安全生产的直接责任人，乙方应建立健全各项安全生产规章制度；乙方必须严格执行甲方的有关安全生产的规定、制度；施工现场安全管理按照谁施工谁负责的原则。由乙方全面负责施工其期间的安全管理，乙方必须严格遵守安全生产法律、法规、标准、安全生产规章制度和操作规程，熟练掌握事故防范措施和事故应急处理预案；必须按操作规程等有关规定进行作业，严禁在作业现场发生违法乱纪等行为。甲方管理人员有权制止乙方人员违章作业，并根据情节酌情给予处罚；甲方有权对安全意识差、不听安全生产指挥的乙方人员责令退场。11.1.1乙方应建立安全生产责任制度、教育培训制度、检查制度、隐患排查治理制度、设施设备安全管理制度、特种作业人员管理制度等各项制度健全。乙方应对其在养护场地的工作人员进行安全教育，并对他们的安全负责，按国家规定办理相关保险。如发生安全事故，或者造成乙方工作人员或给他人造成人身或财产损害的，均由乙方承担责任并支付全部费用。11.1.2责任制落实：部门、岗位安全生产职责明确，并签订责任书。11.1.3建立健全安全组织机构和专职人员名册、安全生产工作部署和会议记录、安全生产检查记录、安全生产经费台账、消防设施配备及管理、特种作业持证上岗、安全生产应急救援预案等。11.1.4企业资质和现场管理：资质证件齐全，施工现场建立用火、用电、使用易燃材料等各项消防安全责任制度和操作规程，机械设备承租单位与出租单位签订租赁合同和安全管理协议，挖掘、移栽、吊装苗木使用起重机械现场管理。11.2乙方应遵守建设部、北京市建委和其他有关单位关于绿化养护建设安全生产有关管理规定，严格按安全标准组织施工，并随时接受行业安全检查人员依法实施的监督检查，针对本养护特点，尤其是绿化景观大树吊装等，采取必要的安全护措施，消除事故隐患。由于乙方安全措施不力造成事故的责任和因此发生的费用由乙方承担。11.3乙方应对其在施工场地的工作人员进行安全教育，并对他们的安全负责，按国家规定办理相关保险。施工中如发生安全事故，由乙方承担责任并支付全部费用。11.4为确保养护安全施工，乙方将与甲方签署安全生产协议，本合同条款未尽事宜，双方在安全生产协议中进一步明确，与本合同具有同等效力。11.5乙方应自行购置和更新施工安全防护用具及设施，改善安全生产条件和作业环境。如果乙方未按合同的约定落实安全防护措施而导致事故发生，或未及时落实文明施工要求致使政府相关管理部门对此进行通报批评或罚款或导致居民投诉，乙方应承担全部责任，且甲方有权解除合同。11.6乙方应对上岗工作人员进行岗前培训及安全防护等方面的培训，乙方及乙方工作人员在从事委托任务期间所发生的人身损害及安全事故等，以及给他人造成的人身及财产损害等，均由乙方承担全部责任；如乙方未及时处理，造成相关人员上访、围堵政府或有关单位时，或者甲方认为会给甲方造成不利影响的，甲方选择（但无责）先行垫付时，无论甲方垫付是否合理，甲方均有权在应付乙方款项中扣除，不足部分要求乙方进行赔偿，并有权要求乙方按甲方垫付部分的30％承担违约责任。第十二条事故处理12.1发生重大伤亡及其他安全事故，乙方应按有关规定立即上报各有关部门，并通知甲方，同时按政府有关部门要求处理。12.2甲方、乙方对事故责任有争议时，应按政府有关部门认定处理。第十三条特别约定13.1乙方承诺：绝不因甲方逾期支付合同价款而延期或不履行合同义务，甲方逾期支付合同价款不作为乙方逾期不履行、不完全履行合同义务的理由。13.2合同期间，如乙方擅自撤离工作现场或工作人员，或者擅自停工超过3天，或者擅自撤场或停工后经甲方通知在通知期限内仍未复工的，均视为乙方以自己的行为不再履行合同，甲方有权解除本合同。13.3因上述原因甲方行使合同解除权时，本合同自甲方向乙方发出合同解除通知时，本合同解除。本合同解除后就未完工部分，甲方有权另行委托第三方进行完成，第三方所收取的费用由乙方承担，甲方有权在应付乙方合同价款中扣除；所余合同价款，如不足以支付第三方所收取的费用，甲方有权向乙方追偿。13.4无论出现任何原因，乙方均不得采取强占工作场地、围堵甲方或政府机关及有关单位，或上街游行，向有关部门上访，在各种载体发表有关言论等，均视为乙方严重违约，甲方有权解除合同。13.5合同经甲方通知解除后，乙方应按甲方的通知要求妥善做好已完养护和已购材料、设备的保护和移交工作，按甲方要求将自有机械设备和人员撤出施工场地。甲方应为乙方撤出提供必要条件，所发生的费用由乙方承担，并按合同约定支付已完养护价款。已经订货的材料、设备由订货方负责退货或解除订货合同，不能退还的货款和因退货、解除订货合同发生的费用，由乙方承担，因未及时退货造成的损失由乙方承担。除此之外，乙方应当赔偿甲方因合同解除给造成的损失。13.6乙方未按照本合同约定的施工节点进行工、完成施工任务，或者未按本合同要求进行施工，以及出现违反本合同约定的有关情形时，在甲方通知要求的期限内仍未完成或改正的，甲方有权解除本合同。13.7无论任何原因乙方未经甲方同意而擅自撤离施工人员的，均视为乙方以自己的行为不再履行合同，本合同自乙方撤离人员之日起终止及解除。13.8如乙方未能向委派到甲方工作的人员发放工资、缴纳社会保险等，或者未能及时妥善解决安全事故等；造成员工或有关人员投诉或上访等，甲方有权解除合同，且甲方有权利用未支付给乙方的合同款项按照甲方核定数额先行支付，并在应付乙方款项中扣除，不足部分甲方可以（但无责）先行垫付并有权向乙方追偿。第十四条违约责任14.1乙方须自行完成本施工任务，严禁乙方将本项目再次进行分包或或将合同项目下无论是全部还是部分权利转让给第三方，否则，乙方应按合同价款的30％承担违约责任，且甲方有权解除合同。14.2若乙方未能按合同约定的内容给甲方提供相应的养护服务或甲方根据合同约定对乙方提出限期整改，但乙方拒不服从或整改不到位，甲方有权解除合同。14.3乙方未按照甲方要求的时间进行整改，每延期一天乙方应按合同总价款的1％向甲方支付逾期完工违约金。逾期5日仍未完成，甲方有权解除本合同。14.4乙方逾期完成本合同约定的养护工作的，每逾期一个日历日应按本合同总价款的1％支付违约金。若乙方逾期完成养护工作达10个日历日以上的，甲方有权单方解除本合同。14.5乙方未履行合同义务，经甲方通知后在通知的期限内仍未履行的，甲方有权解除合同。14.6因乙方原因解除或终止合同的，甲方应向乙方有权拒付未支付合同价款，并要求乙方向甲方支付相当于合同总金额20％的违约金，赔偿甲方的损失。如发生诉讼，乙方还应承担甲方支出的包括但不限于诉讼费、保全费、律师代理费、评估费、拍卖费、办案差旅费等有关费用。14.7甲方不按时支付绿化养护款的违约责任：如甲方无正当理由而故意拖欠绿化养护款或尾款的，则每拖延一日，甲方应按拖欠金额的全国银行间同业拆借中心公布的贷款市场报价利率支付利息，除此之外，甲方无需承担其他违约责任。第十五条通知及送达15.1本合同项下任何一方向对方发出的通知等法律文书，应按本合同记载的地址等联系方式采用直接送达、邮寄送达等，直接送达的以对方签收为有效送达，邮寄送达的以交邮后的第三日视为送达。一方变更通讯地址的应及时书面通知对方当事人，对方当事人实际收到变更通知前的送达仍为有效送达。15.2本合同记载的地址等联系方式为双方工作联系往来、法律文书及争议解决时人民法院和／或仲裁机构的法律文书送达地址，人民法院和／或仲裁机构的诉讼文书（含裁判文书）向任何合同任何一方当事人的上述地址和／或工商登记公示地址（居民身份证登记地址）送达的，视为有效送达。15.3合同送达条款与争议解决条款均为独立条款，不受合同整体或其他条款的效力的影响。第十六条争议解决方式本合同在履行过程中发生的争议由双方当事人协商解决，协商不成的，依法向合同履行地大兴区人民法院起诉；第十七条合同生效及终止17.1本合同自双方法定代表人或授权代表签字并加盖公章之日起生效。17.2合同份数本合同一式陆份，甲方执肆份，乙方执贰份，具有同等法律效力。第十八条其他约定18.1双方根据有关法律、行政法规规定，结合本绿化养护实际，经协商一致后，可对本合同具体化、补充或修改。18.2乙方必须按照北京市相关规定缴纳农民工工伤保险并报甲方备案，上述保险费用已包括在合同总价中。18.3如施工过程中出现扰民、民扰、拖欠雇佣的民工工资等与本绿化养护有关的问题，由乙方全权负责。18.4乙方为实施该项目所需的办公管理用房和管理人员宿舍及劳务人员的居住场所，均由乙方自行解决，甲方不予提供。18.5北京市《城市园林绿化养护管理标准》特级进行养护、《城市园林绿化养护管理规范》（DB11／T213-2014）一级标准及《日常养护管理实施细则》、《大兴区园林服务中心管理考核工作方案》，以及甲方的管理制度等，均是本合同的有效组成部分，乙方应当遵照执行。．6本合同未尽事宜，双方另行签订补充协议，补充协议是本合同的组成部分，与本合同具有同等效力。（以下无正文，签章页）甲方单位：（盖章）乙方单位：（盖章）法定代表人授权代表保托法定代表人或授权代表：（签字或盖章）：董强地址：地址：邮政编码：102600邮政编码：电话：60217734电话：开户行：开户行：账号：签订日期：2021年12月31日签订日期：2021年12月31日附件：1．《绿化养护服务项目廉政责任书》"
    res = model.predict(text)
    for k, v in res.items():
        print(k, v)
