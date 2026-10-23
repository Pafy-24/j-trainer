"""Compatibility imports for the shared managed-channel runtime."""
from jester_sdk_runtime.channel_sessions import (
    ChannelSessionDenied,
    ManagedChannelContext,
    ManagedChannelGuard,
    rm_introspector,
)

__all__ = [
    "ChannelSessionDenied", "ManagedChannelContext", "ManagedChannelGuard",
    "rm_introspector", "guard_from_managed_environment",
]


def guard_from_managed_environment() -> ManagedChannelGuard:
    return ManagedChannelGuard(ManagedChannelContext.from_environment())
