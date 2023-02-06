#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/12/13 10:08
# @Author  : Yan Kaifeng
# @Site    : 
# @File    : model_paths.py
# @Software: PyCharm


# crf_model_path = '/szzn/langming/code/contract_review/train_log/v2/9_20_first_stage_500_short_entities_output/bert/checkpoint-7000'
# span_model_path = '/szzn/langming/code/contract_review/train_log/v2/9_20_first_stage_500_long_entities_output/bert/checkpoint_11000-loss_0.007574732683731385-f1_0.635820895522388/'
import os

PORT = 13240  # 5042    # 线上13240

# dev环境
# MODEL_ROOT = '/data/nlp_models/contract_review/v2'

# 线上环境
cur_file_path = os.path.dirname(os.path.split(os.path.realpath(__file__))[0])
MODEL_ROOT = os.path.join(cur_file_path, 'online_model')

# 11-27日前线上模型路径
# crf_model_path = os.path.join(MODEL_ROOT, 'short_entities_model_2021_10_9')
# span_model_path = os.path.join(MODEL_ROOT, 'long_entities_model_2021_09_27')

# crf_model_path = os.path.join(MODEL_ROOT, '11_4_short_510_down_sampling_8422_adversarial_focal')

# crf_model_path = os.path.join(MODEL_ROOT, '12_6_models',
#                               'short_checkpoint_8000-loss_6.1991473539375965-f1_0.9204114780725501')  # precision: 0.8639 - recall: 0.8800 - f1: 0.8719 1-27 69盲测集
# 0.8743494423791821 0.8980526918671249 0.8860425692220758 100盲测集

# crf_model_path = os.path.join(MODEL_ROOT, '1_26_2022_models',
#                               'short_checkpoint_10000-loss_9.075235140496407-f1_0.9322493224932249')  # precision: 0.8558 - recall: 0.8679 - f1: 0.8618 1-27数据集
#                                                                                                       0.8646222887060584 0.8827796869033983 0.8736066502928396 1-24 100份盲测集

# crf_model_path = os.path.join(MODEL_ROOT, '3_1_2022_models',
#                               'short_checkpoint_10000-loss_7.009158050784698-f1_0.9484536082474226')  # 0.9088531938737393 0.9289805269186713 0.9188066465256799

crf_model_path = os.path.join(MODEL_ROOT, '3_30_2022_models',
                              'short_checkpoint_10000-loss_4.123147506184048-f1_0.9501984572755187')

# crf_model_path = os.path.join(MODEL_ROOT, '3_21_2022_models',
#                               'short_checkpoint_12000-loss_2.9821884115537007-f1_0.9503287649528639')

# crf_model_path = os.path.join(MODEL_ROOT, '5_6_2022_models',
#                               'short_checkpoint_11000-loss_2.855815505439585-f1_0.9560261563381127')
# crf_model_path = '/app_name/nlp/langming/code/contract_review/train_log/2022_3_29_logs/single/3_29_short_augment_data_single_task/augment_data_adversarial_focal_output/bert/checkpoint_14000-loss_3.8890988950376157-f1_0.9417965514644981'
# crf_model_path = '/app_name/nlp/langming/code/contract_review/train_log/2022_3_15_sigle_logs/3_15_long_single_task/split_long_data_focal_adversarial_output/bert/checkpoint_57000-loss_0.0017241084671599353-f1_0.8465361005553932'
# crf_model_path = '/app_name/nlp/langming/code/contract_review/train_log/2022_3_29_logs/single/3_29_long_source_data_single_task/split_short_data_adversarial_focal_output/bert/checkpoint_10000-loss_4.123147506184048-f1_0.9501984572755187' # 0.9259399924041017 0.930889652539137 0.9284082254379283
# crf_model_path = '/app_name/nlp/langming/code/contract_review/train_log/2022_3_29_logs/single/3_29_short_augment_data_single_task__tt/augment_data_adversarial_focal_output/bert/checkpoint_9000-loss_5.6831635877490045-f1_0.9451728247914183' # 0.9259399924041017 0.930889652539137 0.9284082254379283

# crf_model_path = os.path.join(
#     '/app_name/nlp/langming/code/contract_review/train_log/2022_4_12_logs/single/4_12_short_source_data_single_task/split_short_data_adversarial_focal_output/bert/checkpoint_10000-loss_3.514925809739863-f1_0.9478272183190216')
# crf_model_path = os.path.join(
#     '/app_name/nlp/langming/code/contract_review/train_log/2022_4_28_logs/single/4_12_short_source_data_single_task/split_short_data_adversarial_focal_output/bert/checkpoint_11000-loss_2.855815505439585-f1_0.9560261563381127/')  # 0.9268849961919269 0.9293623520427644 0.9281220209723546

# crf_model_path = os.path.join('/szzn/langming/code/contract_review/train_log/v2/12_3_2021_crf/12_3_2021_short_data_adversarial_focal_output/bert', 'checkpoint_8000-loss_6.1991473539375965-f1_0.9204114780725501')
# crf_model_path = os.path.join(
#     '/szzn/langming/code/contract_review/train_log/v2/12_11_2021_crf/short_split_data_12_10_ds_adversarial_focal_output/bert',
#     'checkpoint_9000-loss_6.03793656198602-f1_0.9306968342998383')

# crf_model_path = '/app_name/nlp/langming/code/contract_review/train_log/1_22_logs/1_22_2022_cut_all_data_short_add_data/split_cut_retain_entities_adversarial_focal_output/bert/checkpoint_10000-loss_9.075235140496407-f1_0.9322493224932249'

# crf_model_path = '/app_name/nlp/langming/code/contract_review/train_log/1_22_logs/1_22_2022_cut_all_data_short/split_short_cut_retain_entities_adversarial_focal_output/bert/checkpoint_10000-loss_8.761814262658831-f1_0.9291503766568131'

# span_model_path = os.path.join(MODEL_ROOT, '12_6_models',
#                                'long_checkpoint_49500-loss_0.002037766250485897-f1_0.8175881584675665')

# span_model_path = os.path.join(MODEL_ROOT, '1_26_2022_models',
#                                'long_checkpoint_96000-loss_0.0021110379576796956-f1_0.8190333458224055')

# span_model_path = os.path.join(MODEL_ROOT, '3_1_2022_models',
#                                'long_checkpoint_102000-loss_0.0017387308810023752-f1_0.848870505886096')
#
# span_model_path = os.path.join(MODEL_ROOT, '3_21_2022_models',
#                                'long_checkpoint_99500-loss_0.002060547910184766-f1_0.8395490026019079')

span_model_path = os.path.join(MODEL_ROOT, '3_30_2022_models',
                               'long_checkpoint_30000-loss_0.0012274035388366601-f1_0.8318681318681318')

# span_model_path = os.path.join(MODEL_ROOT, '5_6_2022_models',
#                                'long_checkpoint_72500-loss_0.0014343990187194412-f1_0.8368421052631578')

# span_model_path = '/app_name/nlp/langming/code/contract_review/train_log/2022_4_12_logs/single/4_12_long_source_data_single_task/split_long_data_focal_adversarial_output/bert/checkpoint_103500-loss_0.001535627118502385-f1_0.8353628835849977/' #0.8 0.7503526093088858 0.7743813682678312
# span_model_path = '/app_name/nlp/langming/code/contract_review/train_log/2022_4_28_logs/single/4_28_long_source_data_single_task/split_long_data_focal_adversarial_output/bert/checkpoint_72500-loss_0.0014343990187194412-f1_0.8368421052631578/'  # 0.7888730385164051 0.7799717912552891 0.7843971631205673

# span_model_path = '/app_name/nlp/langming/code/contract_review/train_log/2022_2_15_logs/2_15_2022_cut_all_data_long/split_cut_retain_entities_focal_adversarial_output/bert/checkpoint_106000-loss_0.0020407838153883935-f1_0.8392921632358252'  # 0.7973273942093542 0.7732181425485961 0.7850877192982457 1-27盲测集69份
# 0.7964601769911505 0.7605633802816901 0.7780979827089337 2-24盲测集100份

# span_model_path = '/app_name/nlp/langming/code/contract_review/train_log/2022_2_19_logs/2_19_2022_cut_all_data_long/split_cut_retain_entities_focal_adversarial_output/bert/checkpoint_102000-loss_0.0017387308810023752-f1_0.848870505886096'  # precision: 0.8074 - recall: 0.7970 - f1: 0.8022 1-27盲测集69份
# 0.7985185185185185 0.7591549295774648 0.7783393501805055    2-24盲测集100份

# span_model_path = '/szzn/langming/code/contract_review/train_log/1_13_logs/1_13_2022_cut_all_data_long/down_sampling_0.3/down_sampling_0.3_focal_adversarial_output/bert/checkpoint_70000-loss_0.001842206175429956-f1_0.8231159703240922'
# span_model_path = '/szzn/langming/code/contract_review/train_log/1_13_logs/1_13_2022_cut_all_data_long/split_long_cut_entities_focal_adversarial_output/bert/checkpoint_80000-loss_0.0018325254923938748-f1_0.8256172839506174'
# span_model_path = '/szzn/langming/code/contract_review/train_log/1_13_logs/1_13_2022_cut_all_data_long_ds_0.5/down_sampling_focal_adversarial_output_ds0.5/bert/checkpoint_84000-loss_0.001917883562847157-f1_0.818041634541249'

# span_model_path = '/szzn/langming/code/contract_review/train_log/1_13_logs/1_13_2022_keep_all_data_long/split_long_keep_entities_focal_adversarial_output/bert/checkpoint_64000-loss_0.0016829742483507673-f1_0.815884476534296'
# span_model_path = '/szzn/langming/code/contract_review/train_log/1_17_logs/1_17_2022_keep_long_ds_0.3/down_sampling_0.3_focal_adversarial_output/bert/checkpoint_34000-loss_0.0014440558293839474-f1_0.8184233835252436'
# span_model_path = '/szzn/langming/code/contract_review/train_log/1_17_logs/1_17_2022_keep_long_ds_0.5/down_sampling_0.5_focal_adversarial_output/bert/checkpoint_82000-loss_0.0019105381245579738-f1_0.8139637649138312'

# span_model_path = '/app_name/nlp/langming/code/contract_review/train_log/1_22_logs/1_22_2022_cut_all_data_long/split_long_cut_retain_entities_focal_adversarial_output/bert/checkpoint_96000-loss_0.0021110379576796956-f1_0.8190333458224055'  # precision: 0.8192 - recall: 0.7749 - f1: 0.7964
# span_model_path = '/app_name/nlp/langming/code/contract_review/train_log/1_22_logs/1_22_2022_cut_del_all_data_long/split_long_cut_del_entities_focal_adversarial_output/bert/checkpoint_80000-loss_0.0021054002963569207-f1_0.7996389891696751'  #precision: 0.8234 - recall: 0.6558 - f1: 0.7301
# span_model_path = '/app_name/nlp/langming/code/contract_review/train_log/1_22_logs/1_22_2022_cut_ds_0.3_long/down_sampling_0.3_focal_adversarial_output/bert/checkpoint_98000-loss_0.00201249834407112-f1_0.81889178133135'   #precision: 0.8114 - recall: 0.7727 - f1: 0.7916
# span_model_path = '/app_name/nlp/langming/code/contract_review/train_log/1_22_logs/1_22_2022_cut_ds_0.5_long/down_sampling_0.5_focal_adversarial_output/bert/checkpoint_78000-loss_0.0020209918229257126-f1_0.8154914139568872'  # precision: 0.7905 - recall: 0.7922 - f1: 0.7914

# span_model_path = '/szzn/langming/code/contract_review/train_log/v2/12_23_logs/12_23_2021_ds/down_sampling_focal_adversarial_output/bert/checkpoint_55000-loss_0.0019397225513574797-f1_0.8099963248805585'
# span_model_path = '/szzn/langming/code/contract_review/train_log/v2/12_28_logs/12_28_2021_all_data/split_long_entities_focal_adversarial_output/bert/checkpoint_46500-loss_0.0016881436537140005-f1_0.8034408602150537'  # 切分时，与线上一致
# span_model_path = '/szzn/langming/code/contract_review/train_log/v2/12_23_logs/12_23_2021_all_data/split_long_entities_focal_adversarial_output/bert/checkpoint_76500-loss_0.0017793575485885449-f1_0.8161630870040044'
# span_model_path = os.path.join('/szzn/langming/code/contract_review/train_log/v2/12_3_2021_long/12_3_2021_long_data_focal_adversarial_output/bert', 'checkpoint_49500-loss_0.002037766250485897-f1_0.8175881584675665')
# span_model_path = os.path.join('/data/nlp_models/contract_review/v2/12_6_models/long_checkpoint_49500-loss_0.002037766250485897-f1_0.8175881584675665')
# span_model_path = os.path.join('/szzn/langming/code/contract_review/train_log/v2/12_16_logs/12_16_2021_ds/down_sampling_focal_adversarial_output/bert/checkpoint_41000-loss_0.0018873544346131945-f1_0.808714918759232')
# span_model_path = os.path.join('/szzn/langming/code/contract_review/train_log/v2/12_16_logs/12_16_2021_all_data/split_long_entities_focal_adversarial_output/bert/checkpoint_73500-loss_0.0020480471363209583-f1_0.811529933481153')

#
# span_model_path = os.path.join(
#     '/szzn/langming/code/contract_review/train_log/v2/12_11_2021_long/long_split_data_12_10_ds_focal_adversarial_output/bert/',
#     'checkpoint_45000-loss_0.0019105697224924357-f1_0.8144367480860372')
