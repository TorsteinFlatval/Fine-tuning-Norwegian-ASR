import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
from datasets import load_dataset
import re

#Dirctory for saving model
repo_name = "/path/to/save_model"

#Loading NB_samtale dataset
data_nb_samtale =  load_dataset("Sprakbanken/nb_samtale", cache_dir = "/localhome/studenter/torstefl/huggingface_cache")
print(data_nb_samtale)
unwanted_substring = "/nn/"

#Removing nynorsk, and  0.3s > audios 21s from the dataset
def filtered_small_dataset(example):
    audio_path = example['audio']['path']
    duration = example['duration']
    return unwanted_substring not in audio_path and 21 > duration > 0.3
for split in data_nb_samtale.keys():
    data_nb_samtale[split] = data_nb_samtale[split].filter(filtered_small_dataset)

    # Remove unwanted characters

def replace_chars(example):
    replacements = {'!': '"', '%':"", "'": '',',':'"','-':"",'.':"",'?':"",'_':"",'£':'','é':"",'è':"",'ò':"",'ó':"",'ô':"",'ü':"",'"':"" }  # Update this with your characters
    example['transcription'] = re.sub(r'\b\w+\\', '', example['transcription'])
    for old_char, new_char in replacements.items():
        example['transcription'] = example['transcription'].replace(old_char, new_char)
    example['transcription'] = example['transcription'].lower()

    return example

data_nb_samtale = data_nb_samtale.map(replace_chars)

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("NbAiLab/nb-wav2vec2-300m-bokmaal", cache_dir = "/localhome/studenter/torstefl/huggingface_cache")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("NbAiLab/nb-wav2vec2-300m-bokmaal", cache_dir = "/localhome/studenter/torstefl/huggingface_cache")
processor = Wav2Vec2Processor.from_pretrained("NbAiLab/nb-wav2vec2-300m-bokmaal", cache_dir = "/localhome/studenter/torstefl/huggingface_cache",feature_extractor=feature_extractor, tokenizer=tokenizer)

#pre-processing the dataset
def prepare_dataset(batch):

    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcription"]).input_ids
    return batch

data_nb_samtale = data_nb_samtale.map(prepare_dataset, remove_columns=data_nb_samtale.column_names["train"], num_proc=4)


#creating the dataclass

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor.from_pretrained("NbAiLab/nb-wav2vec2-300m-bokmaal", cache_dir = "/localhome/studenter/torstefl/huggingface_cache",feature_extractor=feature_extractor, tokenizer=tokenizer)
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
from evaluate import load

data_collator = DataCollatorCTCWithPadding(processor=processor, padding = True)
word_error_rate = load("wer")

#use word error rate to compute metrics
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = word_error_rate.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

model = Wav2Vec2ForCTC.from_pretrained("NbAiLab/nb-wav2vec2-300m-bokmaal", cache_dir = "/localhome/studenter/torstefl/huggingface_cache", ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id)
model.freeze_feature_encoder()

from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir= repo_name,
  group_by_length=True,
  per_device_train_batch_size= 1,
  per_device_eval_batch_size = 8,
  evaluation_strategy="steps",
  num_train_epochs=20,
  fp16=True,
  gradient_checkpointing=False, 
  save_steps=1724,
  eval_steps=1724,
  logging_steps=1724,
  learning_rate=1e-4,
  weight_decay=0.005,
  warmup_steps=1000,
  save_total_limit=2,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=data_nb_samtale["train"],
    eval_dataset=data_nb_samtale["validation"],
    tokenizer=processor.feature_extractor,
)

trainer.train()
trainer.save_model(repo_name)
tokenizer.save_pretrained(repo_name)

