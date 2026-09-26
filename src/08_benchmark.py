import re
from collections import defaultdict

from datasets import load_dataset

dataset = load_dataset(
    "GEM/e2e_nlg",
    trust_remote_code=True,
)

sample = dataset["validation"][0]

mr = sample["meaning_representation"]


def parse_mr(mr):
    pairs = re.findall(r"([^,\[]+)\[([^\]]+)\]", mr)

    result = {}

    for slot, value in pairs:
        slot = slot.strip()
        value = value.strip()

        result[slot] = value

    return result


parsed_mr = parse_mr(mr)

print("Original MR:")
print(mr)

print("\nParsed MR:")
print(parsed_mr)

slot_values = defaultdict(set)

for split in ["train", "validation", "test"]:
    for sample in dataset[split]:
        parsed = parse_mr(sample["meaning_representation"])

        for slot, value in parsed.items():
            slot_values[slot].add(value)


print("\n========== Slot Vocabulary ==========\n")

for slot in sorted(slot_values.keys()):
    print(f"{slot}:")
    for value in sorted(slot_values[slot]):
        print(f"  - {value}")
    print()


VALUE_PATTERNS = {
    "area": {
        "city centre": [
            "city centre",
            "city center",
        ],
        "riverside": [
            "riverside",
            "river side",
        ],
    },
    "customer rating": {
        "low": [
            "low rating",
            "low rated",
            "poor rating",
        ],
        "average": [
            "average rating",
            "average rated",
            "average customer rating",
        ],
        "high": [
            "high rating",
            "high rated",
            "high customer rating",
        ],
        "1 out of 5": [
            "1 out of 5",
            "one out of five",
            "1 star",
            "one star",
        ],
        "3 out of 5": [
            "3 out of 5",
            "three out of five",
            "3 star",
            "3-star",
            "three star",
            "three-star",
        ],
        "5 out of 5": [
            "5 out of 5",
            "five out of five",
            "5 star",
            "5-star",
            "five star",
            "five-star",
        ],
    },
    "eatType": {
        "coffee shop": [
            "coffee shop",
        ],
        "pub": [
            "pub",
        ],
        "restaurant": [
            "restaurant",
        ],
    },
    "familyFriendly": {
        "yes": [
            "family friendly",
            "family-friendly",
            "suitable for families",
        ],
        "no": [
            "not family friendly",
            "not family-friendly",
            "not suitable for families",
        ],
    },
    "food": {
        "Chinese": ["chinese"],
        "English": ["english"],
        "Fast food": ["fast food"],
        "French": ["french"],
        "Indian": ["indian"],
        "Italian": ["italian"],
        "Japanese": ["japanese"],
    },
    "priceRange": {
        "cheap": [
            "cheap",
        ],
        "high": [
            "high priced",
            "high-priced",
            "expensive",
        ],
        "less than £20": [
            "less than £20",
            "under £20",
            "below £20",
        ],
        "moderate": [
            "moderate",
            "moderately priced",
        ],
        "more than £30": [
            "more than £30",
            "over £30",
            "above £30",
        ],
        "£20-25": [
            "£20-25",
            "£20 to £25",
            "between £20 and £25",
        ],
    },
}


def normalize(text):
    return text.lower().strip()


def value_is_expressed(slot, value, output):
    output = normalize(output)

    # open-set entity
    if slot in ["name", "near"]:
        return normalize(value) in output

    patterns = VALUE_PATTERNS.get(slot, {}).get(value, [])

    return any(normalize(pattern) in output for pattern in patterns)

# ============================================================
# 检测模型输出实际表达了哪些 slot-value
# ============================================================

def detect_realized_slots(output):
    output = normalize(output)

    realized = {}

    # 1. open-set slot：name / near
    # 直接从数据集里已经统计好的实体值中查
    for slot in ["name", "near"]:
        detected = []

        for value in slot_values[slot]:
            if normalize(value) in output:
                detected.append(value)

        if detected:
            realized[slot] = detected

    # 2. closed-set categorical slot
    for slot, value_patterns in VALUE_PATTERNS.items():
        detected = []

        # familyFriendly 有一个特殊问题：
        # "not family friendly" 本身包含 "family friendly"
        # 所以优先检测 no
        if slot == "familyFriendly":
            no_patterns = value_patterns["no"]

            if any(normalize(p) in output for p in no_patterns):
                realized[slot] = ["no"]
                continue

        for value, patterns in value_patterns.items():
            if any(normalize(p) in output for p in patterns):
                detected.append(value)

        if detected:
            realized[slot] = detected

    return realized

# ============================================================
# 比较 MR 和模型实际表达内容
# ============================================================

def evaluate_one_sample(mr, output):
    expected = parse_mr(mr)
    realized = detect_realized_slots(output)

    correct = {}
    missing = {}
    wrong = {}
    added = {}

    # 1. MR 本来要求表达的 slot
    for slot, expected_value in expected.items():

        # 完全没检测到这个 slot
        if slot not in realized:
            missing[slot] = expected_value
            continue

        detected_values = realized[slot]

        # 检测到了正确值，而且没有冲突值
        if (
            expected_value in detected_values
            and len(detected_values) == 1
        ):
            correct[slot] = expected_value

        else:
            wrong[slot] = {
                "expected": expected_value,
                "detected": detected_values,
            }

    # 2. MR 根本没有要求，但模型自己表达出来了
    for slot, detected_values in realized.items():
        if slot not in expected:
            added[slot] = detected_values

    return {
        "correct": correct,
        "missing": missing,
        "wrong": wrong,
        "added": added,
    }

# ============================================================
# 转换成 Precision / Recall / F1 / SER
# ============================================================

def calculate_metrics(result):
    tp = len(result["correct"])

    # wrong 既意味着：
    # 该说的正确事实没有忠实表达 -> FN
    # 又表达了错误事实 -> FP
    fn = len(result["missing"]) + len(result["wrong"])
    fp = len(result["added"]) + len(result["wrong"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    total_expected = tp + fn

    ser = (
        len(result["missing"])
        + len(result["wrong"])
        + len(result["added"])
    ) / total_expected if total_expected > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "slot_error_rate": ser,
    }

test_output = (
    "The riverside restaurant near Raja Indian Cuisine "
    "has a high rating and is family friendly."
)

result = evaluate_one_sample(mr, test_output)
metrics = calculate_metrics(result)

print("\n========== Evaluation ==========\n")
print("Output:")
print(test_output)

print("\nCorrect:")
print(result["correct"])

print("\nMissing:")
print(result["missing"])

print("\nWrong:")
print(result["wrong"])

print("\nAdded:")
print(result["added"])

print("\nMetrics:")
print(metrics)