from .._core import builtin, tensor, _unwrap_if_constexpr

__all__ = ["pin_vgpr", "pin_lds_address", "ra_hint"]


@builtin
def pin_vgpr(x, base, _semantic=None):
    """Place each thread's elements of `x` in fixed VGPRs.

    Elements go in register order, one 32-bit or two 16-bit elements per VGPR,
    in groups of 8 VGPRs: one gfx1250 WMMA fragment or accumulator tile each.
    With an int `base` the groups follow each other from `base`, so a tensor
    with 128 fp32 elements per thread occupies v[base:base+127]. With a list,
    group i starts at `base[i]`. The values are unchanged; the request applies
    where the call is written.

    Args:
        x (tensor): The tensor whose registers to place.
        base (int or list[int]): The first VGPR, or the first VGPR of each group.
    """
    base = _unwrap_if_constexpr(base)
    bases = [_unwrap_if_constexpr(b) for b in base] if isinstance(base, (list, tuple)) else [base]
    if not bases or any(not isinstance(b, int) or b < 0 for b in bases):
        raise ValueError(f"base must be a non-negative int or a list of them, got {base!r}")
    handle = _semantic.builder.create_amd_pin_vgpr(x.handle, bases)
    return tensor(handle, x.type)


@builtin
def pin_lds_address(x, first, count, _semantic=None):
    """Take the LDS address registers of the loads producing `x` from VGPRs
    [first, first + count).

    `x` must be the result of a shared-memory load. Loads whose addresses
    differ by a constant share one register; the backend splits them so that
    each constant fits the DS offset field. Returns `x` unchanged.

    Args:
        x (tensor): The loaded tensor.
        first (int): The first VGPR of the range.
        count (int): The number of VGPRs in the range.
    """
    first = _unwrap_if_constexpr(first)
    count = _unwrap_if_constexpr(count)
    if not isinstance(first, int) or first < 0 or not isinstance(count, int) or count <= 0:
        raise ValueError(f"first and count must be non-negative ints, got {first!r}, {count!r}")
    handle = _semantic.builder.create_amd_pin_lds_address(x.handle, first, count)
    return tensor(handle, x.type)


@builtin
def ra_hint(x, window=-1, group=0, priority=100, guard=0, _semantic=None):
    """Tell the backend's register allocation what `x`'s registers are for.

    Unlike `pin_vgpr`, nothing is placed here: the backend (gfx1250 partial
    VGPR allocation, after scheduling) uses these facts together with the
    exact schedule. The registers are taken in groups of 8 VGPRs.

    Args:
        x (tensor): The tensor.
        window (int or list[int]): The 256-VGPR window (0-3) for all groups, or
            one per group; -1 for no preference.
        group (int): Tensors with the same non-zero group share a window.
        priority (int): Weight of this tensor's conflicts, percent (100 = default).
        guard (int): Instructions to wait before reusing its registers after
            their last read (0 = backend default).
    """
    window = _unwrap_if_constexpr(window)
    windows = [_unwrap_if_constexpr(w) for w in window] if isinstance(window, (list, tuple)) else [window]
    if not windows or any(not isinstance(w, int) or w < -1 or w > 3 for w in windows):
        raise ValueError(f"window must be -1..3 or a list of them, got {window!r}")
    args = [_unwrap_if_constexpr(a) for a in (group, priority, guard)]
    if any(not isinstance(a, int) or a < 0 for a in args):
        raise ValueError(f"group, priority and guard must be non-negative ints, got {args!r}")
    handle = _semantic.builder.create_amd_ra_hint(x.handle, windows, *args)
    return tensor(handle, x.type)
