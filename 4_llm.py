from be_great import GReaT
from sklearn.datasets import fetch_california_housing
import pandas as pd
# data = fetch_california_housing(as_frame=True).frame
data = pd.read_csv("./DS/train_shap_52.csv")

model = GReaT(llm='gpt2', batch_size=8, epochs=50, save_steps=400000)
model.fit(data)
# model.save("./trainer_great/gpt2")  # saves a "model.pt" and a "config.json" file

synthetic_data = model.sample(n_samples=5000,max_length=1000000000)
synthetic_data.to_csv("./DS/llm_generated_ds.csv", index=False)
