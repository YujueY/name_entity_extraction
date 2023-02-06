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
            'roberta': (BertConfig, BertSpanForNer, CNerTokenizer),
        }

        # self.label_list = ["O", "合同变更的条件", "合同生效条件", "合同终止的条件", "支付方式", "法律依据", "解决方式"]
        self.label_list = [
                "O", "文件标题", "项目编号", "项目名称", "采购方式", "委托有效期开始时间", "文件签发日期", "供应商资质",
                "付款条件描述", "落款名称", "成交供应商", "合同名称", "总金额", "合同价格条款", "付款比例",
                "质保金", "合同签订时间", "乙方违约金条款", "验收方式"
            ]
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
                R = bert_extract_item_v2(start_logits[:len(txt) + 2], end_logits[:len(txt) + 2])
                # R = bert_extract_item_v2(start_logits, end_logits)
                # txt = txt.replace("\x00", "")
                txt_length = len(txt)
                for x in R:
                    results[self.id2label[x[0]]].append([txt[x[1]:x[2] + 1], pre_length + x[1], pre_length + x[2] + 1])
                pre_length += txt_length
        # print("batch_size", "*" * 10, self.args.batch_size)
        new_res = {}
        temp = []
        for k, v in results.items():
            for vv in v:
                temp.append([k, vv[1], vv[2]-1])
        new_res['entities'] = temp
        return [new_res]

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

        # logger.info("Creating features")
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
        # if ex_index % 10000 == 0:
        #     logger.info("Writing example %d of %d", ex_index, len(examples))
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
    checkpoint = "/app_name/nlp/azun/supertext_auto/data/exp/model/yanyi-zaozhuangbiaoshu/chinese_roberta_L-4_H-512_span_枣庄标书+_v4"
    model = GeneralModelSpanPredict(max_seq_length=max_seq_length, checkpoint=checkpoint)
    text = "从专家库中确定评标委员会成员。3评标委员会负责对各投标文件进行评审、比较、评定，并确定中标人。4招标单位代表参加开标会议四、评标程序1宣布评标纪律以及回避提示；2组织推荐评标委员会组长；3资格性审查；4符合性审查；5技术评审；6澄清有关问题；7比较与评价；8确定中标人；9编"
    print(text)
    res = model.predict(text)
    print(res)
    for k, v in res.items():
        print(k, v)
