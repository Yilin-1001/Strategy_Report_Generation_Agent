"""Save merged version"""
from pathlib import Path
import os
from rag_project.data_loader.text_line_merger import TextLineMerger

base = Path("知识库/知识库")

for root, dirs, files in os.walk(base):
    for file in files:
        if file.endswith('.txt') and '2021' in file:
            full_path = Path(root) / file

            if 'backup' in file.lower() or 'cleaned' in file.lower():
                continue

            print("Found:", full_path.name)

            with open(full_path, 'r', encoding='utf-8') as f:
                text = f.read()

            print("Chars:", len(text))
            print("Lines:", len(text.splitlines()))

            merger = TextLineMerger()
            merged, stats = merger.merge_with_stats(text)

            print("Merged chars:", len(merged))
            print("Merged lines:", len(merged.splitlines()))
            print("Reduced:", stats['lines_reduced'])

            output = full_path.parent / f"{full_path.stem}_merged.txt"
            with open(output, 'w', encoding='utf-8') as f:
                f.write(merged)

            print("Saved:", output.name)
            print("Location:", output.parent)
