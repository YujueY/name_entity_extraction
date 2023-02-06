# -*- coding:utf-8 -*-
'''
Author       : azun
Date         : 2022-05-30 16:24:31
LastEditors  : azun
LastEditTime : 2022-05-30 16:25:34
'''
import argparse

def get_argparse():
    parser = argparse.ArgumentParser()
    # Required parameters

    parser.add_argument("--model_type", default="roberta", type=str,
                        help="Model type selected in the list: ")

    # Other parameters
    parser.add_argument('--markup', default='bios', type=str,
                        choices=['bios', 'bio'])
    parser.add_argument('--loss_type', default='ce', type=str,
                        choices=['lsr', 'focal', 'ce'])
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name", )
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    # adversarial training
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit", )
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--device", type=str, default="cpu", help="For distant debugging.")
    return parser