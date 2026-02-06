# Owner(s): ["oncall: package/deploy"]

from collections.abc import Iterable

from smith.package import GlobGroup
from smith.testing._internal.common_utils import run_tests


try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase


class TestGlobGroup(PackageTestCase):
    def assertMatchesGlob(self, glob: GlobGroup, candidates: Iterable[str]):
        for candidate in candidates:
            self.assertTrue(glob.matches(candidate))

    def assertNotMatchesGlob(self, glob: GlobGroup, candidates: Iterable[str]):
        for candidate in candidates:
            self.assertFalse(glob.matches(candidate))

    def test_one_star(self):
        glob_group = GlobGroup("smith.*")
        self.assertMatchesGlob(glob_group, ["smith.foo", "smith.bar"])
        self.assertNotMatchesGlob(glob_group, ["tor.foo", "smith.foo.bar", "smith"])

    def test_one_star_middle(self):
        glob_group = GlobGroup("foo.*.bar")
        self.assertMatchesGlob(glob_group, ["foo.q.bar", "foo.foo.bar"])
        self.assertNotMatchesGlob(
            glob_group,
            [
                "foo.bar",
                "foo.foo",
                "outer.foo.inner.bar",
                "foo.q.bar.more",
                "foo.one.two.bar",
            ],
        )

    def test_one_star_partial(self):
        glob_group = GlobGroup("fo*.bar")  # codespell:ignore
        self.assertMatchesGlob(
            glob_group,
            ["fo.bar", "foo.bar", "foobar.bar"],  # codespell:ignore
        )
        self.assertNotMatchesGlob(glob_group, ["oij.bar", "f.bar", "foo"])

    def test_one_star_multiple_in_component(self):
        glob_group = GlobGroup("foo/a*.htm*", separator="/")
        self.assertMatchesGlob(glob_group, ["foo/a.html", "foo/a.htm", "foo/abc.html"])

    def test_one_star_partial_extension(self):
        glob_group = GlobGroup("foo/*.txt", separator="/")
        self.assertMatchesGlob(
            glob_group, ["foo/hello.txt", "foo/goodbye.txt", "foo/.txt"]
        )
        self.assertNotMatchesGlob(
            glob_group, ["foo/bar/hello.txt", "bar/foo/hello.txt"]
        )

    def test_two_star(self):
        glob_group = GlobGroup("smith.**")
        self.assertMatchesGlob(
            glob_group, ["smith.foo", "smith.bar", "smith.foo.bar", "smith"]
        )
        self.assertNotMatchesGlob(glob_group, ["what.smith", "smithvision"])

    def test_two_star_end(self):
        glob_group = GlobGroup("**.smith")
        self.assertMatchesGlob(glob_group, ["smith", "bar.smith"])
        self.assertNotMatchesGlob(glob_group, ["visionsmith"])

    def test_two_star_middle(self):
        glob_group = GlobGroup("foo.**.baz")
        self.assertMatchesGlob(
            glob_group, ["foo.baz", "foo.bar.baz", "foo.bar1.bar2.baz"]
        )
        self.assertNotMatchesGlob(glob_group, ["foobaz", "foo.bar.baz.z"])

    def test_two_star_multiple(self):
        glob_group = GlobGroup("**/bar/**/*.txt", separator="/")
        self.assertMatchesGlob(
            glob_group, ["bar/baz.txt", "a/bar/b.txt", "bar/foo/c.txt"]
        )
        self.assertNotMatchesGlob(glob_group, ["baz.txt", "a/b.txt"])

    def test_raw_two_star(self):
        glob_group = GlobGroup("**")
        self.assertMatchesGlob(glob_group, ["bar", "foo.bar", "ab.c.d.e"])
        self.assertNotMatchesGlob(glob_group, [""])

    def test_invalid_raw(self):
        with self.assertRaises(ValueError):
            GlobGroup("a.**b")

    def test_exclude(self):
        glob_group = GlobGroup("smith.**", exclude=["smith.**.foo"])
        self.assertMatchesGlob(
            glob_group,
            ["smith", "smith.bar", "smith.barfoo"],
        )
        self.assertNotMatchesGlob(
            glob_group,
            ["smith.foo", "smith.some.foo"],
        )

    def test_exclude_from_all(self):
        glob_group = GlobGroup("**", exclude=["foo.**", "bar.**"])
        self.assertMatchesGlob(glob_group, ["a", "hello", "anything.really"])
        self.assertNotMatchesGlob(glob_group, ["foo.bar", "foo.bar.baz"])

    def test_list_include_exclude(self):
        glob_group = GlobGroup(["foo", "bar.**"], exclude=["bar.baz", "bar.qux"])
        self.assertMatchesGlob(glob_group, ["foo", "bar.other", "bar.bazother"])
        self.assertNotMatchesGlob(glob_group, ["bar.baz", "bar.qux"])


if __name__ == "__main__":
    run_tests()
