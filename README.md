# Fine-tuning-Norwegian-ASR
A fine-tuned model of National Libraries Wav2vec 2.0 model
The model can be found: https://huggingface.co/Tflatval/wav2vec2-nb-finetuned/tree/main
Using the model run this:

...
from transformers import Wav2Vec2Processor, AutoModelForCTC

processor = Wav2Vec2Processor.from_pretrained("Tflatval/wav2vec2-nb-finetuned", cache_dir = "/localhome/studenter/torstefl/huggingface_cache")
model = AutoModelForCTC.from_pretrained("Tflatval/wav2vec2-nb-finetuned", cache_dir = "/localhome/studenter/torstefl/huggingface_cache")
...
