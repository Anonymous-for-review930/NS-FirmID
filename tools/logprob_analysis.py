import math
import json
from typing import List, Dict


def analyze_logprobs(token_logprobs: List[Dict]) -> Dict:
    """
    分析 token logprobs，提供详细的统计信息

    Args:
        token_logprobs: 包含 logprob 的 token 列表

    Returns:
        统计信息字典
    """
    if not token_logprobs:
        return {"error": "Empty logprobs list"}

    logprobs = [item.get("logprob", 0) for item in token_logprobs if item.get("logprob") is not None]

    if not logprobs:
        return {"error": "No valid logprobs found"}

    # 转换为概率
    probabilities = [math.exp(lp) for lp in logprobs]

    # 统计分析
    analysis = {
        "total_tokens": len(logprobs),
        "logprob_stats": {
            "min": min(logprobs),
            "max": max(logprobs),
            "mean": sum(logprobs) / len(logprobs),
            "median": sorted(logprobs)[len(logprobs) // 2]
        },
        "probability_stats": {
            "min": min(probabilities),
            "max": max(probabilities),
            "mean": sum(probabilities) / len(probabilities),
            "median": sorted(probabilities)[len(probabilities) // 2]
        },
        "confidence_levels": {
            "very_high (p > 0.99)": sum(1 for p in probabilities if p > 0.99),
            "high (0.9 < p <= 0.99)": sum(1 for p in probabilities if 0.9 < p <= 0.99),
            "medium (0.7 < p <= 0.9)": sum(1 for p in probabilities if 0.7 < p <= 0.9),
            "low (p <= 0.7)": sum(1 for p in probabilities if p <= 0.7)
        },
        "perplexity": calculate_perplexity(logprobs)
    }

    return analysis


def calculate_perplexity(logprobs: List[float]) -> float:
    """
    计算困惑度 (Perplexity)
    困惑度越低，模型越确定
    """
    if not logprobs:
        return float('inf')

    avg_log_prob = sum(logprobs) / len(logprobs)
    perplexity = math.exp(-avg_log_prob)
    return perplexity


def interpret_logprob(logprob: float) -> str:
    """
    解释单个 logprob 值的含义
    """
    if logprob is None:
        return "无效值"

    prob = math.exp(logprob)

    if prob > 0.99:
        confidence = "非常高"
    elif prob > 0.9:
        confidence = "高"
    elif prob > 0.7:
        confidence = "中等"
    elif prob > 0.5:
        confidence = "较低"
    else:
        confidence = "低"

    return f"概率: {prob:.6f} ({prob * 100:.4f}%), 置信度: {confidence}"


def display_token_analysis(token_logprobs: List[Dict], top_n: int = 10):
    """
    展示每个 token 的详细分析
    """
    print("\n" + "=" * 80)
    print("Token 详细分析 (前{}个)".format(top_n))
    print("=" * 80)

    for i, item in enumerate(token_logprobs[:top_n]):
        token = item.get("decoded_token", item.get("token", ""))
        logprob = item.get("logprob")

        print(f"\nToken {i + 1}: '{token}'")
        print(f"  Logprob: {logprob}")
        if logprob is not None:
            print(f"  {interpret_logprob(logprob)}")


def check_common_issues(token_logprobs: List[Dict]):
    """
    检查常见的异常情况
    """
    issues = []

    # 检查是否有 logprob = 0.0 的情况
    zero_logprobs = sum(1 for item in token_logprobs
                        if item.get("logprob") == 0.0)
    if zero_logprobs > 0:
        issues.append(f"发现 {zero_logprobs} 个 logprob=0.0 的token (概率=100%)")

    # 检查是否有异常大的负值
    large_negative = sum(1 for item in token_logprobs
                         if item.get("logprob") is not None and item.get("logprob") < -10)
    if large_negative > 0:
        issues.append(f"发现 {large_negative} 个 logprob<-10 的token (概率<0.005%)")

    # 检查是否有 None 值
    none_values = sum(1 for item in token_logprobs
                      if item.get("logprob") is None)
    if none_values > 0:
        issues.append(f"发现 {none_values} 个缺失的 logprob 值")

    return issues


# 使用示例
if __name__ == "__main__":
    # 模拟一些 token logprobs 数据
    example_logprobs = [
        {"decoded_token": "The", "logprob": -2.074220174108632e-05},
        {"decoded_token": " quick", "logprob": -0.15},
        {"decoded_token": " brown", "logprob": -0.8},
        {"decoded_token": " fox", "logprob": 0.0},
        {"decoded_token": " jumps", "logprob": -1.5},
        {"decoded_token": ".", "logprob": -0.01},
    ]

    print("示例分析:")
    print("=" * 80)

    # 显示详细分析
    display_token_analysis(example_logprobs)

    # 整体统计
    print("\n" + "=" * 80)
    print("整体统计信息")
    print("=" * 80)
    stats = analyze_logprobs(example_logprobs)
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # 检查问题
    print("\n" + "=" * 80)
    print("异常检查")
    print("=" * 80)
    issues = check_common_issues(example_logprobs)
    if issues:
        for issue in issues:
            print(f"⚠️  {issue}")
    else:
        print("✓ 未发现异常")

    print("\n" + "=" * 80)
    print("关键指标解释")
    print("=" * 80)
    print("""
    • Perplexity (困惑度): 越低越好，表示模型越确定
      - < 10: 非常好
      - 10-50: 良好
      - 50-100: 一般
      - > 100: 较差

    • Logprob 范围:
      - 接近 0: 模型非常确定 (好)
      - -1 到 -5: 模型比较确定 (正常)
      - < -10: 模型不确定 (可能有问题)

    • 特殊值:
      - logprob = 0.0: 概率为 100%，可能是特殊token或数值截断
      - logprob 接近 0: 模型非常有信心
    """)