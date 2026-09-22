from datasets import load_dataset

# 加载E2E数据集
dataset = load_dataset(
    "GEM/e2e_nlg",
    trust_remote_code=True,
)

print(dataset)  # 打印数据集对象的结构概览
"""
Train：训练，直接参与梯度更新。
Validation：训练过程中测试，用来指导训练过程和超参数选择，但不直接参与梯度更新。
Test：训练完成后的最终测试。
Challenge：额外的挑战性评测集，通常从普通数据里构造/抽取，
    用来专门检验模型的 robustness（鲁棒性） 和 generalization（泛化性），
    不一定只是“更难”，而是更针对某些能力做压力测试。
"""

print("\n--- one sample ---")
sample = dataset["train"][0]  # 类似字典的训练集第0条样本

print(sample)
"""
{
    "gem_id": "e2e_nlg-train-0",
    "gem_parent_id": "e2e_nlg-train-0",
    "meaning_representation": "name[The Eagle],
                                eatType[coffee shop],
                                food[Japanese],
                                priceRange[less than £20],
                                customer rating[low],
                                area[riverside],
                                familyFriendly[yes],
                                near[Burger King]",
    "target": "The Eagle is a low rated coffee shop near Burger King and the riverside that is family friendly and is less than £20 for Japanese food.",
    "references": [],
}
    gem_id          = 这条数据是谁
    gem_parent_id   = 这条数据从谁来的
    MR              = 给模型的条件/事实，大致包括 8 类，例如：name, near, area, food, eatType, priceRange,customer rating, familyFriendly
    target          = 希望模型生成的一种答案
    references      = 其他也正确的参考答案
"""

print("\nMR:")
print(sample["meaning_representation"])  # meaning_representation 语义表示

print("\nTarget:")
print(sample["target"])