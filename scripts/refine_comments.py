"""
Comment Refinement Script
=========================
Automatically refine Vietnamese comments to English and simplify overly detailed explanations.

This script helps convert AI-generated style comments to natural developer comments.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


# Patterns to simplify or translate
REFINEMENTS = [
    # Remove overly detailed math explanations
    (r'Toán \([^)]+\):\s+[^"]+?(?=Args:|Returns:|""")', ''),
    (r'Kiến thức Toán:\s+[^"]+?(?=OOP|Args:|""")', ''),
    
    # Simplify Vietnamese to English common phrases
    ('Lưu danh sách', 'Store list of'),
    ('Tính toán', 'Compute'),
    ('Khởi tạo', 'Initialize'),
    ('Trả về', 'Return'),
    ('Kiểm tra', 'Check'),
    ('Chuyển đổi', 'Convert'),
    ('Cập nhật', 'Update'),
    ('Xóa', 'Delete'),
    ('Thêm', 'Add'),
    
    # Remove redundant inline comments
    (r'\s*#\s*Lấy\s+', '  # Get '),
    (r'\s*#\s*Tính\s+', '  # Calculate '),
    (r'\s*#\s*Clone để không thay đổi tensor gốc', ''),
    (r'\s*#\s*\d+% cơ hội áp dụng', ''),
]


def refine_comments(content: str) -> str:
    """Apply refinement patterns to simplify comments."""
    result = content
    
    for pattern, replacement in REFINEMENTS:
        if isinstance(pattern, str):
            result = result.replace(pattern, replacement)
        else:
            result = re.sub(pattern, replacement, result, flags=re.DOTALL)
    
    return result


def process_file(filepath: Path) -> bool:
    """Process a single Python file."""
    try:
        content = filepath.read_text(encoding='utf-8')
        refined = refine_comments(content)
        
        if refined != content:
            filepath.write_text(refined, encoding='utf-8')
            print(f"✓ Refined: {filepath}")
            return True
        else:
            print(f"- Skipped: {filepath} (no changes)")
            return False
            
    except Exception as e:
        print(f"✗ Error processing {filepath}: {e}")
        return False


def main():
    """Process all Python files in src/ directory."""
    src_dir = Path("src")
    
    if not src_dir.exists():
        print("Error: src/ directory not found")
        sys.exit(1)
    
    python_files = list(src_dir.rglob("*.py"))
    print(f"\nFound {len(python_files)} Python files\n")
    
    refined_count = 0
    for filepath in python_files:
        if process_file(filepath):
            refined_count += 1
    
    print(f"\n{'='*60}")
    print(f"Refinement Complete!")
    print(f"{'='*60}")
    print(f"Total files: {len(python_files)}")
    print(f"Refined: {refined_count}")
    print(f"Skipped: {len(python_files) - refined_count}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
