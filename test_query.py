import httpx
r = httpx.post('http://127.0.0.1:8000/api/kb-query', json={'query': 'AI', 'top_k': 3}, timeout=120)
print('Status:', r.status_code)
d = r.json()
sc = d.get('scorecard', {})
print(f"Overall: {sc.get('overall_score', 0):.2f} [{sc.get('grade')}]")
print(f"Stages ({len(sc.get('stages', []))}):")
for s in sc.get('stages', []):
    print(f"  {s['name']}: {s['score']:.2f} [{s['grade']}] diag={s.get('diagnosis','')[:30]}")
    if s.get('details'):
        print(f"    details: {s['details']}")
print(f"Retrieval: {len(d.get('retrieval', []))} items")
for it in d.get('retrieval', []):
    print(f"  RRF={it['score']:.4f} bm25={it.get('bm25_score',0):.4f} vec={it.get('vector_score',0):.4f}")
print(f"Hallucination: {d.get('hallucination')}")
print(f"Answer: {d.get('answer','')[:100]}")
