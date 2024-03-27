import os
import random
import numpy as np
import pathlib
from collections import defaultdict
from datasets import load_dataset

from evaluation.builder.data_builder import BuildEvalData
from evaluation.dtos.dto import IntentClassificationInstance, EvalDataIC, \
    SlotFillingInstance, SlotInstance, EvalDataSF

class BuildMassiveEvalData(BuildEvalData):
    """
    MASSIVE 1.1 is a parallel dataset of > 1M utterances across 52 languages with annotations for the Natural Language
    Understanding tasks of intent prediction and slot annotation. Utterances span 60 intents and include 55 slot types.
    MASSIVE was created by localizing the SLURP dataset, composed of general Intelligent Voice Assistant single-shot
    interactions.
    https://huggingface.co/datasets/AmazonScience/massive
    """

    def __init__(self, data_path: str, language: str) -> None:
        self.data_path = data_path
        self.lang = language

        self.test_data = self.get_data(split='test', lang=self.lang)
        self.test_data_100 = self.test_data.shuffle(seed=42).select(range(100))
        self.dev_data = self.get_data(split='validation', lang=self.lang)

        self.train_data = self.get_data(split='train', lang=self.lang)

        self.output_dir = os.path.join(self.data_path, "eval/")
        pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def get_data(self, split: str, lang: str):
        dataset = load_dataset("AmazonScience/massive", lang, split=split)
        return dataset

    def build_eval_data(self):

        domains, intents = self.extract_intents(data=self.train_data)
        self.extract_slots(data=self.train_data)

        self.build_ic_data(data=self.test_data, domains=domains, intents=intents)
        # self.build_ic_data(data=self.test_data_100, output_path='intents_data_100.json', domains=domains, intents=intents)
        ic_train_data = self.build_ic_data(data=self.train_data, output_path='intents_data_train.json', domains=domains, intents=intents)

        self.build_sf_data(data=self.test_data)
        # self.build_sf_data(data=self.test_data_100, output_path='slots_data_100.json', )
        sf_train_data = self.build_sf_data(data=self.train_data, output_path='slots_data_train.json', )

        # self.build_few_shot_ic_data(output_path='few_shot_intents_10.json')

        for k in [10]: #, 20, 50, 100]:
            for i in range(1, 2):    # Build different version of few-shot examples
                self.build_few_shot_sf_data_per_slot(sf_train_data, output_path=f"few_shot_slots_{k}_{i}.json", k_per_slot=k, random_seed=i)

        for k in [10]: #, 20, 50, 100]:
            for i in range(1, 2):  # Build different version of few-shot examples
                self.build_few_shot_ic_data_per_intent(ic_train_data, output_path=f"few_shot_intents_{k}_{i}.json", k_per_intent=k, random_seed=i)

    def build_eval_data_intent_only(self):

        domains, intents = self.extract_intents(data=self.train_data)

        self.build_ic_data(data=self.test_data, domains=domains, intents=intents)
        self.build_ic_data(data=self.test_data_100, output_path='intents_data_100.json', domains=domains, intents=intents)
        ic_train_data = self.build_ic_data(data=self.train_data, output_path='intents_data_train.json', domains=domains, intents=intents)

        self.build_few_shot_ic_data(output_path='few_shot_intents_10.json')

        for k in [10, 20, 50, 100]:
            for i in range(1, 4):  # Build different version of few-shot examples
                self.build_few_shot_ic_data_per_intent(ic_train_data, output_path=f"few_shot_intents_{k}_{i}.json",
                                                       k_per_intent=k, random_seed=i)

    def build_ic_data(self, data, output_path: str = 'intents_data.json', domains=[], intents=[]) -> EvalDataIC:
        gold_data = []

        for row in data:
            domain = data.features['scenario'].int2str(row['scenario'])
            intent = data.features['intent'].int2str(row['intent'])

            gold_data.append(
                IntentClassificationInstance(
                    utterance=row['utt'],
                    domain=domain,
                    intent=intent
                )
            )
            domains.append(domain)
            intents.append(intent)

        domains = sorted(list(set(domains)))
        intents = sorted(list(set(intents)))

        self.save_as_json(
            data=EvalDataIC(data=gold_data, domains=domains, intents=intents).dict(),
            output_path=os.path.join(self.output_dir, output_path)
        )

        print(self.output_dir, output_path, len(gold_data))

        return EvalDataIC(data=gold_data, domains=domains, intents=intents)

    def build_sf_data(self, data, output_path: str = 'slots_data.json') -> EvalDataSF:
        gold_data = []

        import re
        p = re.compile(r'\[(\w+?) : (.+?)\]')

        for row in data:
            slot_annotated = row['annot_utt']
            slot_matches = p.findall(slot_annotated)

            domain = data.features['scenario'].int2str(row['scenario'])
            intent = data.features['intent'].int2str(row['intent'])

            slots = []
            for (slot, value) in slot_matches:
                slots.append(
                    SlotInstance(
                        slot_name=slot,
                        slot_values=[value]
                    )
                )

            gold_data.append(
                SlotFillingInstance(
                    utterance=row['utt'],
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

        for row in data:
            domain = data.features['scenario'].int2str(row['scenario'])
            intent = data.features['intent'].int2str(row['intent'])

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

        #MASSIVE general slots
        general_slots = ["time", "date", "timeofday", "general_frequency",
                         "device_type", "meal_type", "food_type", "business_type", "media_type",
                         "business_name", "event_name", "place_name", "artist_name", "app_name",
                         "person", "relation"]

        import re
        p = re.compile(r'\[(\w+?) : (.+?)\]')

        for row in data:
            slot_annotated = row['annot_utt']
            slot_matches = p.findall(slot_annotated)

            domain = data.features['scenario'].int2str(row['scenario'])
            intent = data.features['intent'].int2str(row['intent'])

            for (slot, value) in slot_matches:
                if slot in general_slots:
                    slots['all'][slot] = slot.replace("_", " ")
                    if slot not in slot_counts['all']: slot_counts['all'][slot] = 0
                    slot_counts['all'][slot] += 1
                else:
                    slots[intent][slot] = slot.replace("_", " ")
                    if slot not in slot_counts[intent]: slot_counts[intent][slot] = 0
                    slot_counts[intent][slot] += 1

                if domain not in slot_labels[slot]: slot_labels[slot][domain] = 0
                slot_labels[slot][domain] += 1

        slot_labels = {slot:{key:val for key, val in slot_labels[slot].items() if val != 1} for slot in slot_labels}

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
            data=EvalDataIC(data=gold_data, domains=[], intents=[]).dict(),
            output_path=os.path.join(self.output_dir, output_path)
        )

        print(self.output_dir, output_path, k_per_intent, (len(gold_data) / len(ic_train_data.data)))

        return EvalDataIC(data=gold_data, domains=[], intents=[])

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
    for lang in ['en-US', 'de-DE', 'fr-FR', 'it-IT', 'es-ES']:
        builder = BuildMassiveEvalData(data_path=f"data/eval/amz_{lang.split('-')[0]}/", language=lang)
        builder.build_eval_data()

