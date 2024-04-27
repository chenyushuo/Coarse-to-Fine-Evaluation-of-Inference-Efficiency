import numpy as np
import pickle
from collections import defaultdict, OrderedDict
import json
from tqdm import tqdm
from torch.autograd import DeviceType
from torch.autograd.profiler_util import FunctionEventAvg, EventList
import argparse
from torch import nn
import ncu_report

name2stack_str = OrderedDict([('LlamaModel',
              'transformers/models/llama/modeling_llama.py(972): forward'),
             ('LlamaDecoderLayer',
              'transformers/models/llama/modeling_llama.py(762): forward'),
             ('LlamaAttention',
              'transformers/models/llama/modeling_llama.py(351): forward'),
             ('LlamaMLP',
              'transformers/models/llama/modeling_llama.py(250): forward'),
             ('RotaryEmbedding',
              'transformers/models/llama/modeling_llama.py(211): apply_rotary_pos_emb'),
             ('Linear', 'torch/nn/modules/linear.py(113): forward'),
             ('KVCache', 'transformers/cache_utils.py(95): update'),
             ('RMSNorm','transformers/models/llama/modeling_llama.py(112): forward')])

def filt(events):
    filted_events = []
    for event in events:
        if not( event.device_type != DeviceType.CUDA or event.name.startswith('Mem')):
            filted_events.append(event)
    return filted_events
def main(args):
    filted_events = []
    with open(args.events_path, 'rb') as f:
        events = pickle.load(f)
    print(len(events))
    events = filt(events)
    event2tag = {}
    tag_list = []
    last_tag = []
    llama_model_cnt = 0
    linear_cnt = 0
    for event in events:
        if event.device_type != DeviceType.CUDA or event.name.startswith('Mem'):
            tag_list.append([])
            event2tag.append([])
            continue
        tag = []
        for key, value in name2stack_str.items():
            if value in event.stack:
                tag.append(key)
        if len(tag) == 0 and len(last_tag) > 0:
            llama_model_cnt += 1
        else:
            if 'LlamaAttention' in tag:
                if 'Linear' in tag:
                    linear_cnt += 1
                    if linear_cnt <= 3:
                        tag[-1] = 'QKV'
                    else:
                        tag[-1] = 'Out'
                        linear_cnt = 0
                elif 'RotaryEmbedding' in tag:
                    pass
                elif 'KVCache' in tag:
                    pass
                else:
                    tag.append('Attention')
            elif 'LlamaMLP' in tag:
                if 'Linear' in tag:
                    linear_cnt += 1
                    if linear_cnt <= 2:
                        tag[-1] = 'GateUp'
                    else:
                        tag[-1] = 'Down'
                        linear_cnt = 0
                else:
                    tag.append('Act')
            else:
                if len(tag) > 0 and 'CUDAFunctor_add' in event.name:
                    tag.append('Add')
        if tag:
            if llama_model_cnt == 0:
                tag_list.append(['Prefill'] + tag)
            else:
                tag_list.append(['Decoding'] + tag)
        else:
            tag_list.append([])
        last_tag = tag
        event2tag[event.id] = tag_list[-1]
        last_tag = tag
    # print(len(event2tag))
    tags2events = {}
    for event in events:
        tags = event2tag[event.id]
        
        name = ""
        for tag in tags:
            name = name  + tag + "_"
        if not tags:
            continue
        # if tags[0]=="Prefill":
        #     print(tags,name)
        tags = name
        # print(tags)
        if tags not in tags2events.keys():
            tags2events[tags] = []
        tags2events[tags].append(event)
    # tags2events["Prefill_LlamaModel_LlamaDecoderLayer_LlamaAttention_QKV_"] = []
    # tags2events["Prefill_LlamaModel_LlamaDecoderLayer_LlamaAttention_RotaryEmbedding_"] = []
    # tags2events["Prefill_LlamaModel_LlamaDecoderLayer_LlamaAttention_Attention_"] = []
    # tags2events["Prefill_LlamaModel_LlamaDecoderLayer_LlamaAttention_Attention_Out_"] = []
    # tags2events["Prefill_LlamaModel_LlamaDecoderLayer_LlamaMLP_"] = []
    print(tags2events.keys())
    for from_key in tags2events.keys():
        for to_key in tags2events.keys():
            if to_key in from_key and to_key != from_key:
                tags2events[to_key] += tags2events[from_key]
    tags2events["Total"] = filt(events)
    # print(len(tags2events))
    filted_events = filt(events)
    # print(len(filted_events))
    ncu_file = ncu_report.load_report(args.ncu_path)
    event2action = {}
    for range_idx in range(ncu_file.num_ranges()):
        current_range = ncu_file.range_by_idx(range_idx)
        now = 0
        for action_idx in range(current_range.num_actions()):    
            action = current_range.action_by_idx(action_idx)
            if action.name() not in filted_events[now].name:
                pass
            else:
                event2action[filted_events[now].id] = action
                now+=1
    
    print(len(tags2events))
    tags2actions = {}
    for tags,events in tags2events.items():
        tags2actions[tags] = []
        for event in filt(events):
            tags2actions[tags].append(event2action[event.id])
    # print(len(tags2actions))
    print(tags2actions.keys())
    def get_IO(actions):
        ans = {}
        for action in actions:
            # print(action.metric_by_name("dram__bytes.sum.per_second").as_double())
            if action.name() not in ans.keys():
                ans[action.name()] = dict()
                ans[action.name()]["gpu__time_duration.sum"] = 0
                ans[action.name()]["dram__bytes_read.sum"] = 0
                ans[action.name()]["dram__bytes_write.sum"] = 0
            ans[action.name()]["gpu__time_duration.sum"] += action.metric_by_name("gpu__time_duration.sum").as_double()
            ans[action.name()]["dram__bytes_read.sum"] += action.metric_by_name("dram__bytes_read.sum").as_double()
            ans[action.name()]["dram__bytes_write.sum"] += action.metric_by_name("dram__bytes_write.sum").as_double()
        temp = []
        for key,values in ans.items():
            # values["dram__bytes_read.sum.per_second"] /= values["gpu__time_duration.sum"]
            # values["dram__bytes_read.sum"] /= 1024*1024*1024
            # values["dram__bytes_write.sum"] /= 1024*1024*1024
            temp.append((key,values["gpu__time_duration.sum"],values["dram__bytes_read.sum"],values["dram__bytes_write.sum"]))
            # temp.append((key,values["gpu__time_duration.sum"]))
        return temp
    def show(mertics):
        tot_read = 0
        tot_write = 0
        tot_time = 0
        for x in mertics:
            print(x)
            tot_time +=x[1]
            tot_read += x[2]
            tot_write += x[3]
        print("total IO: ",tot_write + tot_read)
        # print("total read: ",tot_read)
        print("total time: ",tot_time)
        # print("total write speed: ",tot_write/tot_time,"GB/s")
        # print("total read speed: ",tot_read/tot_time,"GB/s")
    
    for tags,actions in tags2actions.items():
        if tags == '':
            continue
        # if tags in ["Prefill_LlamaModel_LlamaDecoderLayer_LlamaAttention_QKV_","Prefill_LlamaModel_LlamaDecoderLayer_LlamaAttention_RotaryEmbedding_","Prefill_LlamaModel_LlamaDecoderLayer_LlamaAttention_Attention_","Prefill_LlamaModel_LlamaDecoderLayer_LlamaAttention_Attention_Out_","Prefill_LlamaModel_LlamaDecoderLayer_LlamaMLP_"]:
        print(tags)
        show(get_IO(actions))
        print("")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--events_path",
        type=str,
        default=None,
        help="If specified, we will load the events_path to profile.",
    )
    parser.add_argument(
        "--ncu_path",
        type=str,
        default=None,
        help="If specified, we will load the ncu_path to profile.",
    )
    args = parser.parse_args()

    main(args)