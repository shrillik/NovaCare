from datasets import load_dataset

# Hugging Face will download CSV/text version if available
dataset = load_dataset("bitext/Bitext-telco-llm-chatbot-training-dataset", ignore_verifications=True)

print(dataset['train'][0])
import pandas as pd
df = pd.DataFrame(dataset['train'])
df.to_csv("data/bitext_telco_dataset.csv", index=False)
