"""The model-scoped provider uses public registry invalidation, without replacement."""

from unittest.mock import Mock

import pytest

from vllm_fl.dispatch.registry import OpRegistry
from vllm_fl.dispatch.types import BackendImplKind, OpImpl
from vllm_fl.k3 import routing


def test_register_is_idempotent_and_invalidates_cached_dispatch(monkeypatch):
    monkeypatch.setattr(routing, "use_flaggems_op", lambda name: True)
    manager = Mock(registry=OpRegistry())
    monkeypatch.setattr(routing, "get_default_manager", lambda: manager)
    routing.register()
    routing.register()
    assert len(manager.registry.get_implementations("grouped_topk")) == 1
    provider = manager.registry.get_implementation(
        "grouped_topk", "flagos.kimi_k3.grouped_topk"
    )
    assert provider.fn is routing.grouped_topk
    assert provider.kind == BackendImplKind.DEFAULT
    manager.bump_policy_epoch.assert_called_once()


def test_registration_rejects_conflicting_provider(monkeypatch):
    monkeypatch.setattr(routing, "use_flaggems_op", lambda name: True)
    manager = Mock(registry=OpRegistry())
    manager.registry.register_impl(
        OpImpl(
            "grouped_topk",
            "flagos.kimi_k3.grouped_topk",
            BackendImplKind.DEFAULT,
            lambda: None,
        )
    )
    monkeypatch.setattr(routing, "get_default_manager", lambda: manager)
    with pytest.raises(RuntimeError, match="Conflicting K3"):
        routing.register()
    manager.bump_policy_epoch.assert_not_called()


def test_disabled_operator_does_not_register_a_model_override(monkeypatch):
    manager_factory = Mock()
    monkeypatch.setattr(routing, "get_default_manager", manager_factory)
    monkeypatch.setattr(routing, "use_flaggems_op", lambda name: False)
    routing.register()
    manager_factory.assert_not_called()
