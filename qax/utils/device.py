import jax
from collections.abc import Sequence
from typing import TypeVar

from ..core import StateVector, Operator

T = TypeVar('T', StateVector, Operator, jax.Array)


def select_device(device_name: str | None = None) -> jax.Device:
    """
    Selects a single JAX device object by name.
    If multiple devices of the same type are available, it returns the first one.
    """
    if device_name is None:
        return jax.devices()[0]
    try:
        return jax.devices(device_name)[0]
    except IndexError:
        available_devices = jax.devices()
        raise ValueError(
            f"Device '{device_name}' not found. "
            f"Available devices are: {available_devices}"
        )

def list_devices(device_name: str | None = None) -> Sequence[jax.Device]:
    """
    Lists all available JAX device objects of a given name.
    """
    devices = jax.devices(device_name) if device_name else jax.devices()
    if not devices:
        raise ValueError(f"No devices found for backend: '{device_name}'.")
    return devices

def to(obj: T, device: str | jax.Device) -> T:
    """
    Moves a StateVector, Operator, or JAX Array to the specified device.
    """
    if isinstance(device, str):
        target_device = select_device(device)
    else:
        target_device = device

    if isinstance(obj, (StateVector, Operator)):
        data_to_move = obj.data
    else:
        data_to_move = obj

    data_on_new_device = jax.device_put(data_to_move, device=target_device)

    if isinstance(obj, StateVector):
        return StateVector(data_on_new_device)
    elif isinstance(obj, Operator):
        return Operator(data_on_new_device)
    else:
        return data_on_new_device

def select_cpu() -> jax.Device:
    """Selects the CPU device."""
    return select_device('cpu')

def select_gpu() -> jax.Device:
    """Selects the first available GPU device."""
    return select_device('gpu')

def list_gpus() -> Sequence[jax.Device]:
    """Lists a list of all available GPU devices."""
    return list_devices('gpu')