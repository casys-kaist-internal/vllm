import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

# Specify the download directory
DOWNLOAD_DIR = '/mnt/sda/download'

# Load the dataset with cache directory specified
dataset = load_dataset('gbharti/finance-alpaca',
                       cache_dir=DOWNLOAD_DIR)['train']


class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()  # Language modeling task
        }


def distillation_loss(student_logits, teacher_logits, temperature=2.0, alpha=0.5):
    loss_fn = nn.KLDivLoss(reduction='batchmean')
    soft_labels = nn.functional.softmax(teacher_logits / temperature, dim=-1)
    distillation_loss = loss_fn(nn.functional.log_softmax(
        student_logits / temperature, dim=-1), soft_labels)
    return alpha * distillation_loss


class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        outputs = model(**inputs)
        student_logits = outputs.logits
        loss = distillation_loss(student_logits, teacher_logits)
        return (loss, outputs) if return_outputs else loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load pre-trained models with cache directory specified
teacher_model_name = 'facebook/opt-6.7b'
student_model_name = 'facebook/opt-125m'

teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_model_name, cache_dir=DOWNLOAD_DIR).to(device)
student_model = AutoModelForCausalLM.from_pretrained(
    student_model_name, cache_dir=DOWNLOAD_DIR).to(device)

# Load tokenizer with cache directory specified
tokenizer = AutoTokenizer.from_pretrained(
    student_model_name, cache_dir=DOWNLOAD_DIR)

# Set the padding token to the eos_token
tokenizer.pad_token = tokenizer.eos_token

# Prepare dataset
train_dataset = CustomDataset(dataset, tokenizer)


# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="no",
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    fp16=True,  # Enable mixed precision training if you have a compatible GPU
)

# Initialize the trainer
trainer = DistillationTrainer(
    teacher_model=teacher_model,
    model=student_model,
    args=training_args,
    train_dataset=train_dataset,
)

# Train the student model
trainer.train()
