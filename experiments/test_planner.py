"""Test query planner with different query types"""
import os, sys, json
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from financial_rag.retrievers.query_planner import QueryPlanner

planner = QueryPlanner()

queries = [
    "茅台今天收盘价多少",
    "英伟达和华为的AI芯片谁更强",
    "OpenAI的融资历程是怎样的",
    "商汤科技生成式AI业务的前景怎么样",
    "中国AI政策对哪些公司有利好",
]

for q in queries:
    print(f"\n{'='*50}")
    print(f"Q: {q}")
    print(f"{'='*50}")
    plan = planner.plan(q)
    print(f"  Intent:   {plan.intent}")
    print(f"  Strategy: {plan.strategy}")
    print(f"  Simple:   {plan.is_simple}")
    print(f"  Sub-queries ({len(plan.sub_queries)}):")
    for i, sq in enumerate(plan.sub_queries, 1):
        print(f"    [{i}] {sq.query}")
        print(f"        source={sq.source}  mode={sq.mode}")
        print(f"        purpose: {sq.purpose}")

print("\n\nDone.")
