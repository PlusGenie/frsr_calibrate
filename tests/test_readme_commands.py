import re, pathlib


def test_readme_has_no_w_table_mentions():
    readme = pathlib.Path(__file__).parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")
    forbidden = re.findall(r"\bw_table(?:_file)?\b|\buse_tabulated_w\s*=\s*yes\b", text)
    assert not forbidden, f"Found deprecated references: {forbidden}"
