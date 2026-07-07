"""Quick graph quality analysis"""
import xml.etree.ElementTree as ET
import json

GML = "experiments/lightrag_demo/rag_storage/graph_chunk_entity_relation.graphml"
tree = ET.parse(GML)
ns = {"g": "http://graphml.graphdrawing.org/xmlns"}

nodes = tree.findall(".//g:node", ns)
edges = tree.findall(".//g:edge", ns)

# Entity types
types = {}
for n in nodes:
    et = n.find("g:data[@key='d1']", ns)
    t = et.text if et is not None and et.text else "?"
    types[t] = types.get(t, 0) + 1

print(f"Graph: {len(nodes)} nodes, {len(edges)} edges")
print(f"\n=== Entity Types ({len(types)}) ===")
for k, v in sorted(types.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

# Key entities
print(f"\n=== Key Entities (by connections) ===")
degree = {}
for e in edges:
    s, t = e.get("source"), e.get("target")
    degree[s] = degree.get(s, 0) + 1
    degree[t] = degree.get(t, 0) + 1
for name, deg in sorted(degree.items(), key=lambda x: -x[1])[:15]:
    print(f"  {name}: {deg} connections")

# Sample relationships
print(f"\n=== Sample Relationships (first 15) ===")
for e in edges[:15]:
    s, t = e.get("source"), e.get("target")
    kw = e.find("g:data[@key='d9']", ns)
    desc = e.find("g:data[@key='d8']", ns)
    keywords = kw.text[:60] if kw is not None and kw.text else ""
    print(f"  {s} -> {t}")
    if keywords:
        print(f"    keywords: {keywords}")
