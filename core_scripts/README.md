# Core Production Scripts

These are the main production scripts for RAG Pipeline.

## Scripts

### 1. clear_milvus.py
Clear Milvus database before re-indexing.

### 2. chunk_all_documents.py (MAIN)
Process all documents, generate chunks and index to Milvus.

### 3. convert_pdf_to_txt.py
Convert PDF documents to TXT.

### 4. batch_merge_lines.py
Optimize TXT with line merging (reduces 68.6% lines).

### 5. demo_search.py
Interactive search demo for manual testing.

### 6. test_retrieval.py
Automated retrieval testing with report generation.

## Complete Workflow

```bash
# Step 1: Clear database
python clear_milvus.py --yes

# Step 2: Convert PDF (optional)
python convert_pdf_to_txt.py

# Step 3: Line merge (recommended)
python batch_merge_lines.py

# Step 4: Batch indexing
python chunk_all_documents.py

# Step 5: Test search
python demo_search.py
```
