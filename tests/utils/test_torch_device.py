import torch

from carl.utils import torch_device


def test_cuda_disabled_via_env_values(monkeypatch):
    monkeypatch.setenv(torch_device.CUDA_VISIBLE_DEVICES_ENV, "")
    assert torch_device._cuda_disabled_via_env()

    monkeypatch.setenv(torch_device.CUDA_VISIBLE_DEVICES_ENV, " -1 ")
    assert torch_device._cuda_disabled_via_env()

    monkeypatch.setenv(torch_device.CUDA_VISIBLE_DEVICES_ENV, "NONE")
    assert torch_device._cuda_disabled_via_env()

    monkeypatch.setenv(torch_device.CUDA_VISIBLE_DEVICES_ENV, "0")
    assert not torch_device._cuda_disabled_via_env()

    monkeypatch.delenv(torch_device.CUDA_VISIBLE_DEVICES_ENV, raising=False)
    assert not torch_device._cuda_disabled_via_env()


def test_resolve_device_respects_disabled_env(monkeypatch):
    monkeypatch.setenv(torch_device.CUDA_VISIBLE_DEVICES_ENV, "-1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert torch_device.resolve_device(prefer_cuda=True).type == "cpu"


def test_resolve_device_returns_cuda_when_available(monkeypatch):
    monkeypatch.delenv(torch_device.CUDA_VISIBLE_DEVICES_ENV, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert torch_device.resolve_device(prefer_cuda=True).type == "cuda"


def test_resolve_device_falls_back_to_cpu_on_cuda_probe_error(monkeypatch):
    monkeypatch.delenv(torch_device.CUDA_VISIBLE_DEVICES_ENV, raising=False)

    def _raise() -> bool:
        raise RuntimeError("probe failure")

    monkeypatch.setattr(torch.cuda, "is_available", _raise)
    assert torch_device.resolve_device(prefer_cuda=True).type == "cpu"


def test_resolve_device_prefers_cpu_when_requested():
    assert torch_device.resolve_device(prefer_cuda=False).type == "cpu"
