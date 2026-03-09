"""LLVM lit configuration for tiny-ton tests."""

import lit.formats

config.name = "tiny-ton"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".ttn"]
config.test_source_root = os.path.dirname(__file__)
