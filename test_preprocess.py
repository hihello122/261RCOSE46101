"""
preprocess.py 단위 테스트 + 4개 모드별 결과물 미리보기.

conda 환경에서 실행:
    python test_preprocess.py
"""

import tempfile
import shutil
from pathlib import Path

from preprocess import (
    parse_region_file,
    extract_conflict_block,
    add_context,
    has_conflict_markers,
    build_prompt,
    extract_type_context,
    summarize_edit_script,
    process_conflict_pair,
)


# ═══════════════════════════════════════════════════════════════════
# Test fixture
# ═══════════════════════════════════════════════════════════════════

MERGED_CONTENT = """\
package com.example;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class UserService extends BaseService {

    private final UserRepository repo;
    private final Logger logger;

    public List<User> filterActive(List<User> users) {
<<<<<<< a
        List<User> result = new ArrayList<>();
        for (User u : users) {
            if (u.isActive()) {
                result.add(u);
            }
        }
        return result;
||||||| base
        List<User> result = new ArrayList<>();
        for (User u : users) {
            if (u.isEnabled()) {
                result.add(u);
            }
        }
        return result;
=======
        return users.stream()
                .filter(User::isEnabled)
                .collect(Collectors.toList());
>>>>>>> b
    }

    public void saveUser(User user) {
        repo.save(user);
    }
}
"""

RESOLVED_CONTENT = """\
package com.example;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class UserService extends BaseService {

    private final UserRepository repo;
    private final Logger logger;

    public List<User> filterActive(List<User> users) {
        return users.stream()
                .filter(User::isActive)
                .collect(Collectors.toList());
    }

    public void saveUser(User user) {
        repo.save(user);
    }
}
"""

REGION_CONTENT = """\
# (origin_conflict_start, origin_conflict_end, resolved_start, resolved_end)
(13, 30, 13, 16)
"""

# GumTree 출력 시뮬레이션 (ast 모드 테스트용)
FAKE_GUMTREE_BASE_A = """\
Update MethodInvocation [120,135] to [120,132]
  isEnabled() -> isActive()
Update MethodInvocation [120,135] to [120,132]
  isEnabled() -> isActive()
"""

FAKE_GUMTREE_BASE_B = """\
Delete SimpleName [80,86]
Delete Block [87,180]
Delete ForStatement [75,185]
Delete VariableDeclarationStatement [60,74]
Delete ReturnStatement [186,200]
Insert MethodInvocation [60,120]
Insert MethodReference [80,95]
Insert MethodInvocation [96,118]
"""


def make_test_pair(tmp: Path) -> Path:
    """테스트용 conflict pair 디렉토리 생성."""
    pair = tmp / "Java" / "test-project" / "conflict_files_0"
    for subdir in ["merged", "resolved", "regions", "a", "b", "base"]:
        (pair / subdir).mkdir(parents=True, exist_ok=True)

    (pair / "merged" / "UserService.java").write_text(MERGED_CONTENT)
    (pair / "resolved" / "UserService.java").write_text(RESOLVED_CONTENT)
    (pair / "regions" / "UserService.java.region").write_text(REGION_CONTENT)
    (pair / "a" / "UserService.java").write_text("// version a\n")
    (pair / "b" / "UserService.java").write_text("// version b\n")
    (pair / "base" / "UserService.java").write_text("// version base\n")

    return tmp / "Java"


# ═══════════════════════════════════════════════════════════════════
# Unit tests
# ═══════════════════════════════════════════════════════════════════

def test_parse_region_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".region", delete=False) as f:
        f.write("# comment\n(11, 24, 11, 14)\n(30, 40, 28, 35)\n")
        f.flush()
        regions = parse_region_file(f.name)
    assert len(regions) == 2
    assert regions[0] == (11, 24, 11, 14)
    assert regions[1] == (30, 40, 28, 35)
    print("  [PASS] parse_region_file")


def test_extract_conflict_block():
    lines = ["line1\n", "line2\n", "line3\n", "line4\n", "line5\n"]
    assert extract_conflict_block(lines, 2, 4) == "line2\nline3\nline4\n"
    assert extract_conflict_block(lines, 4, 10) == "line4\nline5\n"
    print("  [PASS] extract_conflict_block")


def test_add_context():
    lines = [f"L{i}\n" for i in range(1, 21)]
    before, region, after = add_context(lines, 10, 12, 3)
    assert region == "L10\nL11\nL12\n"
    assert before == "L7\nL8\nL9\n"
    assert after == "L13\nL14\nL15\n"
    before2, _, _ = add_context(lines, 1, 2, 5)
    assert before2 == ""
    print("  [PASS] add_context")


def test_has_conflict_markers():
    assert has_conflict_markers("<<<<<<< a\nfoo\n=======\nbar\n>>>>>>> b\n")
    assert not has_conflict_markers("normal code\n")
    print("  [PASS] has_conflict_markers")


def test_build_prompt_baseline():
    prompt = build_prompt("before\n", "<<<<<<< a\nA\n=======\nB\n>>>>>>> b\n", "after\n")
    assert "// Conflict" in prompt
    assert "// Resolution" in prompt
    assert "Edit script" not in prompt
    assert "Imports" not in prompt
    print("  [PASS] build_prompt (baseline)")


def test_build_prompt_with_ast_and_type():
    prompt = build_prompt(
        "before\n", "conflict\n", "after\n",
        ast_context="base→a: UPDATE MethodInvocation: 2",
        type_context="// Imports\nimport java.util.List;",
    )
    assert "Edit script summary" in prompt
    assert "Imports" in prompt
    assert prompt.index("Imports") < prompt.index("Edit script")
    print("  [PASS] build_prompt (ast+type)")


def test_extract_type_context():
    lines = MERGED_CONTENT.splitlines(keepends=True)
    ctx = extract_type_context(lines, 13, 30)
    assert "java.util.List" in ctx
    assert "UserService" in ctx
    assert "filterActive" in ctx
    print("  [PASS] extract_type_context")


def test_summarize_edit_script():
    summary = summarize_edit_script(FAKE_GUMTREE_BASE_A)
    assert "UPDATE MethodInvocation: 2" in summary
    summary2 = summarize_edit_script(FAKE_GUMTREE_BASE_B)
    assert "DELETE" in summary2
    assert "INSERT MethodInvocation" in summary2
    assert summarize_edit_script("") == "no changes"
    print("  [PASS] summarize_edit_script")


def test_process_conflict_pair_baseline():
    tmp = Path(tempfile.mkdtemp())
    try:
        data_dir = make_test_pair(tmp)
        pair_dir = data_dir / "test-project" / "conflict_files_0"
        samples = process_conflict_pair(pair_dir, "test-project", 5, "baseline", None)
        assert len(samples) == 1
        s = samples[0]
        assert s["mode"] == "baseline"
        assert "<<<<<<<" in s["prompt"]
        assert "ast_context" not in s
        assert "type_context" not in s
        print("  [PASS] process_conflict_pair (baseline)")
    finally:
        shutil.rmtree(tmp)


def test_process_conflict_pair_type():
    tmp = Path(tempfile.mkdtemp())
    try:
        data_dir = make_test_pair(tmp)
        pair_dir = data_dir / "test-project" / "conflict_files_0"
        samples = process_conflict_pair(pair_dir, "test-project", 5, "type", None)
        assert len(samples) == 1
        s = samples[0]
        assert "type_context" in s
        assert "Imports" in s["prompt"]
        print("  [PASS] process_conflict_pair (type)")
    finally:
        shutil.rmtree(tmp)


def test_full_text_format():
    tmp = Path(tempfile.mkdtemp())
    try:
        data_dir = make_test_pair(tmp)
        pair_dir = data_dir / "test-project" / "conflict_files_0"
        samples = process_conflict_pair(pair_dir, "test-project", 3, "baseline", None)
        s = samples[0]
        assert s["text"] == s["prompt"] + s["resolution"]
        print("  [PASS] full text format")
    finally:
        shutil.rmtree(tmp)


# ═══════════════════════════════════════════════════════════════════
# 4개 모드별 결과물 미리보기
# ═══════════════════════════════════════════════════════════════════

def demo_all_modes():
    """baseline / ast / type / ast+type 4가지 모드의 실제 프롬프트 출력 비교."""
    merged_lines = MERGED_CONTENT.splitlines(keepends=True)
    resolved_lines = RESOLVED_CONTENT.splitlines(keepends=True)

    cs, ce, rs, re_ = 13, 30, 13, 16
    context_lines = 5

    before_ctx, conflict_region, after_ctx = add_context(merged_lines, cs, ce, context_lines)
    resolved_region = extract_conflict_block(resolved_lines, rs, re_)
    type_ctx = extract_type_context(merged_lines, cs, ce)

    # GumTree는 실제 바이너리 없이 시뮬레이션
    ast_ctx = (
        f"base→a: {summarize_edit_script(FAKE_GUMTREE_BASE_A)}\n"
        f"base→b: {summarize_edit_script(FAKE_GUMTREE_BASE_B)}"
    )

    modes = {
        "baseline": {"ast_context": None, "type_context": None},
        "ast":      {"ast_context": ast_ctx, "type_context": None},
        "type":     {"ast_context": None, "type_context": type_ctx},
        "ast+type": {"ast_context": ast_ctx, "type_context": type_ctx},
    }

    for mode_name, kwargs in modes.items():
        prompt = build_prompt(before_ctx, conflict_region, after_ctx, **kwargs)
        text = prompt + resolved_region

        print()
        print("█" * 70)
        print(f"  MODE: {mode_name}")
        print("█" * 70)
        print()
        print("─── PROMPT ─────────────────────────────────────────")
        print(prompt)
        print("─── RESOLUTION (label) ─────────────────────────────")
        print(resolved_region)
        print("─── FULL TEXT (prompt + resolution) ─────────────────")
        print(f"  total chars: {len(text)}")
        print(f"  total lines: {len(text.splitlines())}")
        print()


# ═══════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print(" Unit Tests")
    print("=" * 70)

    test_parse_region_file()
    test_extract_conflict_block()
    test_add_context()
    test_has_conflict_markers()
    test_build_prompt_baseline()
    test_build_prompt_with_ast_and_type()
    test_extract_type_context()
    test_summarize_edit_script()
    test_process_conflict_pair_baseline()
    test_process_conflict_pair_type()
    test_full_text_format()

    print("\nAll tests passed!")

    print()
    print("=" * 70)
    print(" Mode Comparison Demo")
    print("=" * 70)

    demo_all_modes()
