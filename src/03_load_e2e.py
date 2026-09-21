from datasets import load_dataset

dataset = load_dataset(
    "GEM/e2e_nlg",
    trust_remote_code=True,
)

print(dataset)

print("\n--- one sample ---")
sample = dataset["train"][0]

print("MR:")
print(sample["meaning_representation"])

print("\nTarget:")
print(sample["target"])