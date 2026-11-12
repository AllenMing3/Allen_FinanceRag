"""
测试治理门禁 meta 测试 — 分级体系自身的回归锁。

治理契约（pytest.ini + tests/conftest.py）：
- 每个用例必须带至少一个执行形态 marker（unit/integration/regression/eval），
  缺失 → 收集期 pytest.UsageError 拒绝（防 216/430 无标记失控重演）。
- critical 是重要性维度（P0 命脉快跑通道），与执行形态正交，不替代门禁。

用 pytester 子进程验证门禁真实生效；find_unmarked 纯函数直接单测。
"""
import pytest

pytestmark = pytest.mark.unit

pytest_plugins = ["pytester"]

_FORMS = ("unit", "integration", "regression", "eval")


class TestFindUnmarked:
    def test_marked_item_passes(self, pytester):
        from tests.conftest import find_unmarked
        pytester.makepyfile("""
            import pytest
            @pytest.mark.unit
            def test_ok():
                assert True
        """)
        items = pytester.getitems(pytester.path)
        assert find_unmarked(items) == []

    def test_unmarked_item_flagged(self, pytester):
        from tests.conftest import find_unmarked
        pytester.makepyfile("""
            def test_bare():
                assert True
        """)
        items = pytester.getitems(pytester.path)
        flagged = find_unmarked(items)
        assert len(flagged) == 1 and "test_bare" in flagged[0]

    def test_slow_or_critical_alone_not_enough(self, pytester):
        """辅助/重要性标签不能替代执行形态：只带 slow/critical 仍被拒。"""
        from tests.conftest import find_unmarked
        pytester.makepyfile("""
            import pytest
            @pytest.mark.slow
            @pytest.mark.critical
            def test_only_aux():
                assert True
        """)
        items = pytester.getitems(pytester.path)
        assert len(find_unmarked(items)) == 1


class TestGateEnforcement:
    _GATE_CONFTEST = """
        import pytest
        _FORMS = ("unit", "integration", "regression", "eval")
        def pytest_collection_modifyitems(config, items):
            bad = [i.nodeid for i in items
                   if not any(i.get_closest_marker(m) for m in _FORMS)]
            if bad:
                raise pytest.UsageError("无执行形态 marker: " + ", ".join(bad))
    """

    def test_gate_rejects_unmarked_collection(self, pytester):
        pytester.makeconftest(self._GATE_CONFTEST)
        pytester.makepyfile("""
            def test_bare():
                assert True
        """)
        result = pytester.runpytest()
        result.assert_outcomes()
        assert result.ret != 0, "无标记用例必须被门禁拒绝收集"
        result.stderr.fnmatch_lines(["*无执行形态 marker*"])

    def test_gate_accepts_marked_collection(self, pytester):
        pytester.makeconftest(self._GATE_CONFTEST)
        pytester.makepyfile("""
            import pytest
            @pytest.mark.unit
            def test_ok():
                assert True
        """)
        result = pytester.runpytest()
        result.assert_outcomes(passed=1)

    def test_real_conftest_gate_wired(self):
        """真实 conftest 必须接线门禁（防 hook 被无意删除）。"""
        from tests import conftest
        assert hasattr(conftest, "pytest_collection_modifyitems")
        assert hasattr(conftest, "find_unmarked")