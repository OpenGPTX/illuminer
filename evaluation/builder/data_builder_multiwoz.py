import os
import json
import random
import pathlib
from typing import List
from collections import defaultdict

from evaluation.builder.data_builder import BuildEvalData
from evaluation.dtos.dto import IntentClassificationInstance, EvalDataIC, \
    SlotFillingInstance, SlotInstance, EvalDataSF


class BuildMultiWozEvalData(BuildEvalData):
    """
    Source: https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2
    """
    def __init__(self, test_data_paths: List[str], dev_data_paths: List[str], train_data_paths: List[str]) -> None:
        self.data_path = 'data/eval/multi_woz/'

        self.test_data = self.get_data(paths=test_data_paths)
        self.dev_data = self.get_data(paths=dev_data_paths)
        self.train_data = self.get_data(paths=train_data_paths)

        self.output_dir = os.path.join(self.data_path, "eval/")
        pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def get_data(self, paths):
        data = []
        for path in paths:
            data.extend(json.load(open(path, 'r')))
        return data

    def build_eval_data(self):
        self.build_ic_data(data=self.test_data)
        self.build_sf_data(data=self.test_data)
        self.build_few_shot_ic_data()
        self.build_few_shot_sf_data()

        ic_train_data = self.build_ic_data(data=self.train_data, output_path='intents_data_train.json')
        sf_train_data = self.build_sf_data(data=self.train_data, output_path='slots_data_train.json')

        self.extract_intents(data=self.train_data)
        self.extract_slots(data=self.train_data)

        for k in [10, 20, 50, 100]:
            for i in range(1, 4):  # Build different version of few-shot examples
                self.build_few_shot_sf_data_per_slot(sf_train_data, output_path=f"few_shot_slots_{k}_{i}.json",
                                                     k_per_slot=k, random_seed=i)

        for k in [10, 20, 50, 100]:
            for i in range(1, 4):  # Build different version of few-shot examples
                self.build_few_shot_ic_data_per_intent(ic_train_data, output_path=f"few_shot_intents_{k}_{i}.json",
                                                       k_per_intent=k, random_seed=i)

    def build_ic_data(self, data, output_path: str = 'intents_data.json') -> EvalDataIC:
        # build the dataset for "intent classification" component
        gold_data = []
        for idx in range(len(data)):
            turn = data[idx]['turns'][0]
            #for turn in data[idx]['turns']:
            speaker = turn['speaker']
            utterance = turn['utterance']
            if speaker == "USER":
                for frame in turn['frames']:
                    if frame['state']['active_intent'] != 'NONE':
                        slot_values = frame['state']['slot_values']
                        if slot_values:
                            domain = frame['service']
                            intent = frame['state']['active_intent']
                            gold_data.append(
                                IntentClassificationInstance(
                                    utterance=utterance,
                                    domain=domain,
                                    intent=intent
                                )
                            )

        self.save_as_json(
            data=EvalDataIC(data=gold_data).dict(),
            output_path=os.path.join(self.output_dir, output_path)

        )

        print(self.output_dir, output_path, len(gold_data))

        return EvalDataIC(data=gold_data)

    def build_sf_data(self, data, output_path: str = 'slots_data.json') -> EvalDataSF:
        # build the dataset for "slot filling" component
        gold_data = []
        for idx in range(len(data)):
            turn = data[idx]['turns'][0]
            #for turn in data[idx]['turns']:
            speaker = turn['speaker']
            utterance = turn['utterance']
            if speaker == "USER":
                for frame in turn['frames']:
                    if frame['state']['active_intent'] != 'NONE':
                        slot_values = frame['state']['slot_values']
                        if slot_values:
                            slots = []
                            for slot in slot_values.keys():
                                slots.append(
                                    SlotInstance(
                                        slot_name=slot,
                                        slot_values=slot_values[slot]
                                    )
                                )
                            domain = frame['service']
                            intent = frame['state']['active_intent']
                            gold_data.append(
                                SlotFillingInstance(
                                    utterance=utterance,
                                    domain=domain,
                                    intent=intent,
                                    slots=slots
                                )
                            )

        self.save_as_json(
            data=EvalDataSF(data=gold_data).dict(),
            output_path=os.path.join(self.output_dir, output_path)
        )

        print(self.output_dir, output_path, len(gold_data))

        return EvalDataSF(data=gold_data)

    def extract_intents(self, data, output_path: str = 'intents_desc.json'):
        intents = defaultdict(defaultdict)
        list_of_intents = set()
        list_of_domains = set()

        for idx in range(len(data)):
            turn = data[idx]['turns'][0]
            speaker = turn['speaker']
            utterance = turn['utterance']
            if speaker == "USER":
                for frame in turn['frames']:
                    if frame['state']['active_intent'] != 'NONE':
                        domain = frame['service']
                        intent = frame['state']['active_intent']

                        intents[domain][intent] = intent.replace("_", " ")
                        list_of_intents.add(intent)
                        list_of_domains.add(domain)

        self.save_as_json(
            data=intents,
            output_path=os.path.join(self.output_dir, output_path)
        )

        return (list(sorted(list_of_domains)), list(sorted(list_of_intents)))

    def extract_slots(self, data, output_path: str = 'slots_desc.json'):
        slots = defaultdict(defaultdict)
        slot_counts = defaultdict(defaultdict)
        slot_labels = defaultdict(defaultdict)

        for idx in range(len(data)):
            turn = data[idx]['turns'][0]
            speaker = turn['speaker']
            utterance = turn['utterance']
            if speaker == "USER":
                for frame in turn['frames']:
                    if frame['state']['active_intent'] != 'NONE':
                        service = frame['service']
                        intent = frame['state']['active_intent']
                        slot_values = frame['state']['slot_values']
                        if slot_values:
                            for slot in slot_values.keys():
                                slots[intent][slot] = slot.replace(f"{service}-", "")

                                if slot not in slot_counts[intent]: slot_counts[intent][slot] = 0
                                slot_counts[intent][slot] += 1

                                if service not in slot_labels[slot]: slot_labels[slot][service] = 0
                                slot_labels[slot][service] += 1

        self.save_as_json(
            data=slots,
            output_path=os.path.join(self.output_dir, output_path)
        )

        self.save_as_json(
            data=slot_counts,
            output_path=os.path.join(self.output_dir, 'slots_count.json')
        )

        self.save_as_json(
            data=slot_labels,
            output_path=os.path.join(self.output_dir, 'slots_labels.json')
        )

    def build_few_shot_ic_data(self, output_path: str = 'few_shot_intents.json') -> EvalDataIC:
        return self.build_ic_data(data=self.dev_data, output_path=output_path)

    def build_few_shot_sf_data(self, output_path: str = 'few_shot_slots.json') -> EvalDataSF:
        return self.build_sf_data(data=self.dev_data, output_path=output_path)

    def build_few_shot_ic_data_per_intent(self, ic_train_data: EvalDataIC, output_path: str = 'few_shot_intents_10.json', k_per_intent: int = 10, random_seed: int = 42) -> EvalDataIC:
        samples_per_intent = {}
        for idx, sample in enumerate(ic_train_data.data):
            if sample.intent not in samples_per_intent: samples_per_intent[sample.intent] = []
            samples_per_intent[sample.intent].append(idx)

        gold_data = []
        for intent in samples_per_intent:
            random.seed(random_seed)
            if k_per_intent < len(samples_per_intent[intent]):
                random_idxs = random.sample(samples_per_intent[intent], k_per_intent)
            else:
                random_idxs = samples_per_intent[intent]
            for idx in random_idxs:
                gold_data.append(ic_train_data.data[idx])

        self.save_as_json(
            data=EvalDataIC(data=gold_data).dict(),
            output_path=os.path.join(self.output_dir, output_path)
        )

        print(self.output_dir, output_path, k_per_intent, (len(gold_data) / len(ic_train_data.data)))

        return EvalDataIC(data=gold_data)

    def build_few_shot_sf_data_per_slot(self, sf_train_data: EvalDataSF, output_path: str = 'few_shot_slots_10.json', k_per_slot: int = 10, random_seed: int = 42) -> EvalDataSF:
        samples_per_slot_name = {}
        for idx, sample in enumerate(sf_train_data.data):
            for slot in sample.slots:
                if slot.slot_name not in samples_per_slot_name: samples_per_slot_name[slot.slot_name] = []
                samples_per_slot_name[slot.slot_name].append(idx)

        counts_per_slot_name = {}
        for slot_name in samples_per_slot_name: counts_per_slot_name[slot_name] = k_per_slot

        gold_data = []
        for slot_name in samples_per_slot_name:
            for count in range(counts_per_slot_name[slot_name]):
                random.seed(random_seed * count)
                random_idx = random.choice(samples_per_slot_name[slot_name])
                random_sample = sf_train_data.data[random_idx]
                gold_data.append(random_sample)

                for slot in random_sample.slots:
                    counts_per_slot_name[slot.slot_name] -= 1

        self.save_as_json(
            data=EvalDataSF(data=gold_data).dict(),
            output_path=os.path.join(self.output_dir, output_path)
        )

        print(self.output_dir, output_path, k_per_slot, (len(gold_data) / len(sf_train_data.data)))

        return EvalDataSF(data=gold_data)


if __name__ == '__main__':
    builder = BuildMultiWozEvalData(
        test_data_paths=['data/eval/multi_woz/test/dialogues_%03d.json' % i for i in range(1, 3)],
        dev_data_paths=['data/eval/multi_woz/dev/dialogues_%03d.json' % i for i in range(1, 3)],
        train_data_paths=['data/eval/multi_woz/train/dialogues_%03d.json' % i for i in range(1, 18)]
    )

    builder.build_eval_data()

