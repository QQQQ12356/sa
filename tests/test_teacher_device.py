"""Teacher placement regression tests; no CUDA hardware or weights needed."""
import os
import unittest
from unittest.mock import patch

from search_comp.utils.teacher_device import resolve_teacher_device


class TeacherDeviceTests(unittest.TestCase):
    def resolve(self, device="auto", env=None, available=True, count=2, current=0):
        with patch.dict(os.environ, env or {}, clear=True), \
             patch("torch.cuda.is_available", return_value=available), \
             patch("torch.cuda.device_count", return_value=count), \
             patch("torch.cuda.current_device", return_value=current):
            return resolve_teacher_device(device)

    def test_two_ranks_use_distinct_logical_gpus(self):
        for mode in ("auto", "cuda"):
            devices = [self.resolve(mode, {"WORLD_SIZE": "2", "LOCAL_RANK": str(rank),
                       "CUDA_VISIBLE_DEVICES": "3,5"}) for rank in range(2)]
            self.assertEqual(devices, ["cuda:0", "cuda:1"])

    def test_rank_one_rejects_fixed_gpu_zero(self):
        with self.assertRaisesRegex(ValueError, "LOCAL_RANK=1"):
            self.resolve("cuda:0", {"WORLD_SIZE": "2", "LOCAL_RANK": "1"})

    def test_explicit_matching_rank(self):
        self.assertEqual(self.resolve("cuda:1", {"WORLD_SIZE": "2", "LOCAL_RANK": "1"}), "cuda:1")

    def test_single_process_current_and_explicit_devices(self):
        self.assertEqual(self.resolve(current=1), "cuda:1")
        self.assertEqual(self.resolve("cuda:1"), "cuda:1")

    def test_explicit_cpu_and_single_process_cpu_fallback(self):
        self.assertEqual(self.resolve("cpu", {"WORLD_SIZE": "2"}, available=False), "cpu")
        self.assertEqual(self.resolve(available=False), "cpu")

    def test_missing_rank_and_out_of_range_fail_early(self):
        with self.assertRaisesRegex(ValueError, "requires LOCAL_RANK"):
            self.resolve(env={"WORLD_SIZE": "2"})
        with self.assertRaisesRegex(ValueError, "visible CUDA devices"):
            self.resolve(env={"WORLD_SIZE": "2", "LOCAL_RANK": "1"}, count=1)

    def test_gpu_request_without_cuda_fails(self):
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            self.resolve("cuda", available=False)
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            self.resolve(env={"WORLD_SIZE": "2", "LOCAL_RANK": "0"}, available=False)

    def test_invalid_configuration(self):
        for device in (None, "mps", "cuda:-1", "cuda:bad"):
            with self.subTest(device=device), self.assertRaisesRegex(ValueError, "distill_teacher_device"):
                self.resolve(device)


if __name__ == "__main__":
    unittest.main()
