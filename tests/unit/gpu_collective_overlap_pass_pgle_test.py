# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests that gpu_collective_overlap_pass looks up fingerprinted PGLE costs."""

import unittest
from unittest import mock

from maxtext.utils import gpu_collective_overlap_pass as pass_lib


def _varint(n: int) -> bytes:
  out = bytearray()
  while True:
    b = n & 0x7F
    n >>= 7
    out.append(b | (0x80 if n else 0))
    if not n:
      return bytes(out)


def _len_field(field: int, payload: bytes) -> bytes:
  return _varint(field << 3 | 2) + _varint(len(payload)) + payload


def _module_proto(attributes: dict[str, str]) -> bytes:
  """A serialized HloModuleProto prefix: name, a large dummy computation, an
  int field, and the frontend attributes."""
  entries = b"".join(_len_field(1, _len_field(1, k.encode()) + _len_field(2, v.encode())) for k, v in attributes.items())
  return (
      _len_field(1, b"jit_train_step")
      + _len_field(3, b"\x00" * 300_000)
      + _varint(5 << 3 | 0)
      + _varint(123456789)
      + _len_field(pass_lib._HLO_MODULE_FRONTEND_ATTRIBUTES_FIELD, entries)  # pylint: disable=protected-access
  )


class PgleFingerprintTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.enterContext(mock.patch.object(pass_lib, "_pgle_costs_by_fingerprint", {}))
    self.enterContext(mock.patch.object(pass_lib, "_profile_costs", {}))
    self.enterContext(mock.patch.object(pass_lib, "_inst_cost_cache", {}))
    self.enterContext(mock.patch.object(pass_lib, "_te_ep_overrides", {}))

  def test_split_cost_name(self):
    self.assertEqual(pass_lib._split_cost_name("4e3c::fusion.1"), ("4e3c", "fusion.1"))  # pylint: disable=protected-access
    self.assertEqual(pass_lib._split_cost_name("fusion.1"), ("", "fusion.1"))  # pylint: disable=protected-access

  def test_module_fingerprint_is_read_from_frontend_attributes(self):
    proto = _module_proto({"other": "x", "fingerprint_before_lhs": "4e3c43a777ad889fa69322ad23ce4c4b"})
    self.assertEqual(pass_lib._module_fingerprint(proto), "4e3c43a777ad889fa69322ad23ce4c4b")  # pylint: disable=protected-access

  def test_module_without_fingerprint(self):
    self.assertIsNone(pass_lib._module_fingerprint(_module_proto({"other": "x"})))  # pylint: disable=protected-access

  def test_truncated_proto_returns_none(self):
    proto = _module_proto({"fingerprint_before_lhs": "abc"})
    self.assertIsNone(pass_lib._module_fingerprint(proto[:1000]))  # pylint: disable=protected-access

  def test_costs_are_selected_by_module_fingerprint(self):
    pass_lib._pool_costs(  # pylint: disable=protected-access
        {
            "aaaa::fusion.1": 10.0,
            "aaaa::reduce-scatter.250": 3000.0,
            "bbbb::fusion.1": 99.0,
            "hand-written.1": 7.0,
        }
    )
    pass_lib._te_ep_overrides["te_ep_dispatch_ffi.3"] = 500.0  # pylint: disable=protected-access
    pass_lib._select_module_costs("aaaa", "m")  # pylint: disable=protected-access
    self.assertEqual(
        pass_lib._profile_costs,  # pylint: disable=protected-access
        {"fusion.1": 10.0, "reduce-scatter.250": 3000.0, "hand-written.1": 7.0, "te_ep_dispatch_ffi.3": 500.0},
    )
    pass_lib._select_module_costs("bbbb", "m")  # pylint: disable=protected-access
    self.assertEqual(pass_lib._profile_costs["fusion.1"], 99.0)  # pylint: disable=protected-access
    self.assertNotIn("reduce-scatter.250", pass_lib._profile_costs)  # pylint: disable=protected-access

  def test_unknown_or_missing_fingerprint_gets_only_unfingerprinted_costs(self):
    pass_lib._pool_costs({"aaaa::fusion.1": 10.0, "fusion.2": 4.0})  # pylint: disable=protected-access
    for fingerprint in ("cccc", None):
      pass_lib._select_module_costs(fingerprint, "m")  # pylint: disable=protected-access
      self.assertEqual(pass_lib._profile_costs, {"fusion.2": 4.0})  # pylint: disable=protected-access


if __name__ == "__main__":
  unittest.main()
