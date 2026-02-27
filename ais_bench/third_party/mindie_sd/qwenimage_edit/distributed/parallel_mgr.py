import os
from typing import List, Optional
from dataclasses import dataclass
import torch.distributed as dist
import torch_npu
import logging
from ais_bench.benchmark.models.local_models.qwenimage_edit.distributed.utils import RankGenerator, generate_masked_orthogonal_rank_groups
from ais_bench.benchmark.models.local_models.qwenimage_edit.distributed.group_coordinator import GroupCoordinator, SequenceParallelGroupCoordinator

#--------- ljf -------------------
import torch
import torch.distributed
try:
    import torch_musa
    from torch_musa.core.device import set_device, device_count
except ModuleNotFoundError:
    pass
#---------------------------

from yunchang import set_seq_parallel_pg
from yunchang.globals import PROCESS_GROUP

_WORLD: Optional[GroupCoordinator] = None
_TP: Optional[GroupCoordinator] = None
_SP: Optional[SequenceParallelGroupCoordinator] = None
_CFG: Optional[GroupCoordinator] = None


@dataclass
class ParallelConfig:
    tp_degree: int = 1
    sp_degree: int = 1
    ulysses_degree: int = 1
    ring_degree: int = 1
    use_cfg_parallel: bool = False
    world_size: int = 1

    def __post_init__(self):
        if self.use_cfg_parallel:
            self.cfg_degree = 2
        else:
            self.cfg_degree = 1
        if not self.tp_degree * self.sp_degree * self.cfg_degree <= self.world_size:
            logging.error(
                "tp_degree * sp_degree * cfg_degree must be less than or equal to "
                "world_size because of classifier free guidance"
            )
        if not (self.world_size % (self.tp_degree * self.sp_degree * self.cfg_degree) == 0):
            logging.error("world_size must be divisible by tp_degree * sp_degree * cfg_degree")


# * QUERY
def get_world_group() -> GroupCoordinator:
    if _WORLD is None:
        logging.error("world group is not initialized")
    return _WORLD


# TP
def get_tp_group() -> GroupCoordinator:
    assert _TP is not None, "tensor model parallel group is not initialized"
    return _TP


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    return get_tp_group().world_size


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    return get_tp_group().rank_in_group


# SP
def get_sp_group() -> SequenceParallelGroupCoordinator:
    if _SP is None:
        logging.error("pipeline model parallel group is not initialized")
    return _SP


def get_sequence_parallel_state():
    """Return state for the sequence parallel group."""
    return _SP is not None


def get_sequence_parallel_world_size():
    """Return world size for the sequence parallel group."""
    if not get_sequence_parallel_state():
        return 1
    return get_sp_group().world_size


def get_sequence_parallel_rank():
    """Return my rank for the sequence parallel group."""
    if not get_sequence_parallel_state():
        return 0
    return get_sp_group().rank_in_group


# CFG
def get_cfg_group() -> GroupCoordinator:
    if _CFG is None:
        logging.error("classifier_free_guidance parallel group is not initialized")
    return _CFG


def get_cfg_state():
    """Return state for the sequence parallel group."""
    return _CFG is not None


def get_classifier_free_guidance_world_size():
    """Return world size for the classifier_free_guidance parallel group."""
    if not get_cfg_state():
        return 1
    return get_cfg_group().world_size


def get_classifier_free_guidance_rank():
    """Return my rank for the classifier_free_guidance parallel group."""
    if not get_cfg_state():
        return 0
    return get_cfg_group().rank_in_group


def init_world_group(
    ranks: List[int], local_rank: int, backend: str
) -> GroupCoordinator:
    return GroupCoordinator(
        group_ranks=[ranks],
        local_rank=local_rank,
        torch_distributed_backend=backend,
    )

# wan2.1 的
def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    distributed_init_method: str = "env://",
    local_rank: int = -1,
    backend: str = "hccl",
):
    logging.debug(
        "world_size=%d rank=%d local_rank=%d " "distributed_init_method=%s backend=%s",
        world_size,
        rank,
        local_rank,
        distributed_init_method,
        backend,
    )
    if not dist.is_initialized():
        if distributed_init_method is None:
            logging.error(
                "distributed_init_method must be provided when initializing "
                "distributed environment"
            )
        # this backend is used for WORLD
        dist.init_process_group(
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank,
        )
    # set the local rank
    # local_rank is not available in torch ProcessGroup,
    # see https://github.com/pytorch/pytorch/issues/122816
    if local_rank == -1:
        # local rank not set, this usually happens in single-node
        # setting, where we can use rank as local rank
        if distributed_init_method == "env://":
            local_rank = int(os.getenv('LOCAL_RANK', 0))
            torch_npu.npu.set_device(local_rank)
        else:
            local_rank = rank
    global _WORLD
    if _WORLD is None:
        ranks = list(range(dist.get_world_size()))
        _WORLD = init_world_group(ranks, local_rank, backend)
    else:
        if not _WORLD.world_size == dist.get_world_size():
            logging.error("world group already initialized with a different world size")


# def init_distributed_environment(
#     world_size: int = -1,
#     rank: int = -1,
#     distributed_init_method: str = "env://",
#     local_rank: int = -1,
#     backend: str = "hccl",
# ):
#     logging.debug(
#         "world_size=%d rank=%d local_rank=%d " "distributed_init_method=%s backend=%s",
#         world_size,
#         rank,
#         local_rank,
#         distributed_init_method,
#         backend,
#     )
#     if not torch.distributed.is_initialized():
#         assert distributed_init_method is not None, (
#             "distributed_init_method must be provided when initializing "
#             "distributed environment"
#         )
#         # this backend is used for WORLD
#         torch.distributed.init_process_group(
#             backend=backend,
#             init_method=distributed_init_method,
#             world_size=world_size,
#             rank=rank,
#         )
#         set_device(torch.distributed.get_rank() % device_count())
#     # set the local rank
#     # local_rank is not available in torch ProcessGroup,
#     # see https://github.com/pytorch/pytorch/issues/122816
#     if local_rank == -1:
#         # local rank not set, this usually happens in single-node
#         # setting, where we can use rank as local rank
#         if distributed_init_method == "env://":
#             # local_rank = int(os.getenv('LOCAL_RANK', 0))
#             local_rank = dist.get_rank()
#             print(f"init_distributed_environment 里面 local_rank {local_rank}")
#         else:
#             local_rank = rank
#     global _WORLD
#     if _WORLD is None:
#         ranks = list(range(torch.distributed.get_world_size()))
#         _WORLD = init_world_group(ranks, local_rank, backend)
#         print(f"_WORLD 初始化")
#     else:
#         assert (
#             _WORLD.world_size == torch.distributed.get_world_size()
#         ), "world group already initialized with a different world size"
#         print(f"_WORLD 没有 初始化")


def model_parallel_is_initialized():
    """Check if tensor and pipeline parallel groups are initialized."""
    return (
        _CFG is not None
        and _SP is not None
        and _TP is not None
    )


def init_model_parallel_group(
    group_ranks: List[List[int]],
    local_rank: int,
    backend: str,
    parallel_mode: str,
    **kwargs,
) -> GroupCoordinator:
    if parallel_mode not in [
        "tensor",
        "sequence",
        "classifier_free_guidance",
    ]:
        logging.error(f"parallel_mode {parallel_mode} is not supported")
    if parallel_mode == "sequence":                # ulysses
        return SequenceParallelGroupCoordinator(
            group_ranks=group_ranks,
            local_rank=local_rank,
            torch_distributed_backend=backend,
            **kwargs,
        )
    else:
        return GroupCoordinator(           #  cfg
            group_ranks=group_ranks,
            local_rank=local_rank,
            torch_distributed_backend=backend,
        )


def initialize_model_parallel(
    classifier_free_guidance_degree: int = 1,
    sequence_parallel_degree: int = 1,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    tensor_parallel_degree: int = 1,
    backend: Optional[str] = None,
) -> None:
    """
    Initialize model parallel groups.

    Arguments:
        classifier_free_guidance_degree: number of GPUs used for Classifier Free Guidance (CFG)
        sequence_parallel_degree: number of GPUs used for sequence parallelism.
        tensor_parallel_degree: number of GPUs used for tensor parallelism.
        backend: distributed backend of pytorch collective comm.
    """
    # Get world size and rank. Ensure some consistencies.
    if not dist.is_initialized():
        logging.error("dist is not initialized")
    world_size: int = dist.get_world_size()
    backend = backend

    if (
        world_size
        != classifier_free_guidance_degree
        * sequence_parallel_degree
        * tensor_parallel_degree
    ):
        raise RuntimeError(
            f"world_size ({world_size}) is not equal to "
            f"sequence_parallel_degree ({sequence_parallel_degree}) x "
            f"classifier_free_guidance_degree "
            f"({classifier_free_guidance_degree}) x "
            f"tensor_parallel_degree "
            f"({tensor_parallel_degree})"
        )
    
    rank_generator: RankGenerator = RankGenerator(
        tensor_parallel_degree,
        sequence_parallel_degree,
        classifier_free_guidance_degree,
        "tp-sp-cfg",
    )
    
    global _CFG
    if _CFG is not None:
        logging.error("classifier_free_guidance group is already initialized")
    _CFG = init_model_parallel_group(
        group_ranks=rank_generator.get_ranks("cfg"),
        local_rank=get_world_group().local_rank,
        backend=backend,
        parallel_mode="classifier_free_guidance",
    )

    global _SP
    if _SP is not None:
        logging.error("sequence parallel group is already initialized")
    set_seq_parallel_pg(
            sp_ulysses_degree=ulysses_degree,
            sp_ring_degree=ring_degree,
            rank=get_world_group().rank_in_group,
            world_size=world_size
        )
    _SP = init_model_parallel_group(
            group_ranks=rank_generator.get_ranks("sp"),
            local_rank=get_world_group().local_rank,
            backend=backend,
            parallel_mode="sequence",
            ulysses_group=PROCESS_GROUP.ULYSSES_PG,
            ring_group=PROCESS_GROUP.RING_PG,
        )

    global _TP
    assert _TP is None, "Tensor parallel group is already initialized"
    _TP = init_model_parallel_group(
        group_ranks=rank_generator.get_ranks("tp"),
        local_rank=get_world_group().local_rank,
        backend=backend,
        parallel_mode="tensor",
    )


def destroy_model_parallel():
    """Set the groups to none and destroy them."""
    global _CFG
    if _CFG:
        _CFG.destroy()
    _CFG = None

    global _SP
    if _SP:
        _SP.destroy()
    _SP = None

    global _TP
    if _TP:
        _TP.destroy()
    _TP = None


def destroy_distributed_environment():
    global _WORLD
    if _WORLD:
        _WORLD.destroy()
    _WORLD = None
    if dist.is_initialized():
        dist.destroy_process_group()


def init_parallel_env(parallel_config: ParallelConfig):
    if not model_parallel_is_initialized():
        logging.warning("Model parallel is not initialized, initializing...")
    init_distributed_environment(
        world_size=dist.get_world_size(),
        rank=dist.get_rank(),
        backend='hccl',
    )
    initialize_model_parallel(
        classifier_free_guidance_degree=parallel_config.cfg_degree,
        sequence_parallel_degree=parallel_config.sp_degree,
        ulysses_degree=parallel_config.ulysses_degree,
        ring_degree=parallel_config.ring_degree,
        tensor_parallel_degree=parallel_config.tp_degree,
    )


def finalize_parallel_env():
    if model_parallel_is_initialized():
        destroy_model_parallel()
    destroy_distributed_environment()
