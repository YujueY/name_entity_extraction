#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/16 11:46
# @Author  : Yan Kaifeng
# @Site    : 
# @File    : ner_multi_crf_span.py
# @Software: PyCharm
import copy
import json
import logging

import torch

logger = logging.getLogger(__name__)


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_start_ids, all_end_ids, all_lens, all_texts = map(
        torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_input_mask = all_input_mask[:, :max_len]
    all_segment_ids = all_segment_ids[:, :max_len]
    return all_input_ids, all_input_mask, all_segment_ids, all_lens, all_texts


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, input_len, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        # self.start_ids = start_ids
        # self.input_len = input_len
        # self.label_ids = label_ids
        # self.end_ids = end_ids
        # self.subjects = subjects

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def convert_examples_to_features(examples, label_dict, max_seq_length, tokenizer,
                                 cls_token_at_end=False, cls_token="[CLS]", cls_token_segment_id=1,
                                 sep_token="[SEP]", pad_on_left=False, pad_token=0, pad_token_segment_id=0,
                                 sequence_a_segment_id=0, mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    crf_label_map = {label: i for i, label in enumerate(label_dict['crf_labels'])}
    span_label_map = {label: i for i, label in enumerate(label_dict['span_labels'])}
    # span_label_map = {label: i for i, label in enumerate(label_list) if label in long_labels}
    # crf_label_map = {label: i for i, label in enumerate(label_list) if label in short_labels}
    features, input_texts = [], []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        textlist = example.text_a
        subjects = example.subject
        tokens = tokenizer.tokenize(textlist)
        start_ids = [0] * len(tokens)
        end_ids = [0] * len(tokens)
        subjects_id = []

        input_texts.append(example.text_a)

        label_ids = [crf_label_map[x] for x in example.labels]  # crf label information

        for subject in subjects:  # span label information
            if subject[0] in span_label_map:
                label = subject[0]
                start = subject[1]
                end = subject[2]
                start_ids[start] = span_label_map[label]
                end_ids[end] = span_label_map[label]
                subjects_id.append((span_label_map[label], start, end))
        # Account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            start_ids = start_ids[: (max_seq_length - special_tokens_count)]
            end_ids = end_ids[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

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
        label_ids += [crf_label_map['O']]
        start_ids += [0]
        end_ids += [0]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [crf_label_map['O']]
            start_ids += [0]
            end_ids += [0]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [crf_label_map['O']] + label_ids
            start_ids = [0] + start_ids
            end_ids = [0] + end_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended t          o.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(label_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            # label_ids = ([pad_token] * padding_length) + label_ids
            # start_ids = ([0] * padding_length) + start_ids
            # end_ids = ([0] * padding_length) + end_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            # label_ids += [pad_token] * padding_length
            # start_ids += ([0] * padding_length)
            # end_ids += ([0] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        # assert len(label_ids) == max_seq_length
        # assert len(start_ids) == max_seq_length
        # assert len(end_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            # logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, input_len=input_len,
                                      segment_ids=segment_ids))
    return features, input_texts
