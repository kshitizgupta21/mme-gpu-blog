from transformers.utils import logging
# set logging to ERROR and above
logging.set_verbosity(40)
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

print("Downloading pretrained DistilBERT Classification TF Model from HuggingFace...")
model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

save_model_dir = '/workspace/hf_distilbert'
print("Serializing Model to SavedModel file...")
model.save_pretrained(save_model_dir, saved_model=True)