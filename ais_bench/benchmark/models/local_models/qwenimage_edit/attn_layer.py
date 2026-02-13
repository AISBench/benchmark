import torch
from torch import Tensor
import torch_npu

import torch.distributed as dist
from yunchang import LongContextAttention
try:
    from yunchang.kernels import AttnType
except ImportError:
    raise ImportError("Please install yunchang 0.6.0 or later")


import math
import os
from typing import Any

from mindiesd import attention_forward



# from yunchang.comm.all_to_all import SeqAllToAll4D
# from yunchang.globals import HAS_SPARSE_SAGE_ATTENTION

from ais_bench.benchmark.models.local_models.qwenimage_edit.distributed.all_to_all import SeqAllToAll4D
import logging

from ais_bench.benchmark.models.local_models.qwenimage_edit.distributed.parallel_mgr import (
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group
)


logger = logging.getLogger(__name__)
MAX_TOKEN = 2147483647


class xFuserLongContextAttention_new4(LongContextAttention):
    ring_impl_type_supported_kv_cache = ["basic"]

    def __init__(
        self,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        ring_impl_type: str = "basic",
        use_pack_qkv: bool = False,
        use_kv_cache: bool = False,
        use_sync: bool = False,
        attn_type: AttnType = AttnType.FA,
        attn_processor: torch.nn.Module = None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    ) -> None:
        """
        Arguments:
            scatter_idx: int = 2, the scatter dimension index for Ulysses All2All
            gather_idx: int = 1, the gather dimension index for Ulysses All2All
            ring_impl_type: str = "basic", the ring implementation type, currently only support "basic"
            use_pack_qkv: bool = False, whether to use pack qkv in the input
            use_kv_cache: bool = False, whether to use kv cache in the attention layer, which is applied in PipeFusion.
            attn_type: AttnType = AttnType.FA, the attention type supported inside long context attention, including "TORCH", "FA", "FA3", "SAGE_FP16", "SAGE_FP8"
            attn_processor: nn.Module = None, the attention processor can be passed in to replace the attention processor if attn_type is do not support it.
        """
        super().__init__(
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            ring_impl_type=ring_impl_type,
            use_pack_qkv=use_pack_qkv,
            use_sync=use_sync,
            attn_type = attn_type,
        )
        self.use_kv_cache = use_kv_cache
        self.q_descale = q_descale
        self.k_descale = k_descale
        self.v_descale = v_descale

        # 校验：仅"basic"类型的环形实现支持KV缓存
        if (
            use_kv_cache
            and ring_impl_type not in self.ring_impl_type_supported_kv_cache
        ):
            raise RuntimeError(
                f"ring_impl_type: {ring_impl_type} do not support SP kv cache."
            )

        self.attn_processor = attn_processor

    @torch.compiler.disable
    def forward(
        self,
        attn,
        query: Tensor,       #  [B, S_image/ulysses_size, H, D]
        key: Tensor,
        value: Tensor,
        *,
        joint_tensor_query=None,         #  [B, S_text, H, D]
        joint_tensor_key=None,
        joint_tensor_value=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        joint_strategy="none",
        txt_pad_len = 0
    ) -> Tensor:
        """forward

        Arguments:
            attn (Attention): the attention module
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args,
            joint_tensor_query: Tensor = None, a replicated tensor among processes appended to the front or rear of query, depends the joint_strategy
            joint_tensor_key: Tensor = None, a replicated tensor among processes appended to the front or rear of key, depends the joint_strategy
            joint_tensor_value: Tensor = None, a replicated tensor among processes appended to the front or rear of value, depends the joint_strategy,
            *args: the args same as flash_attn_interface
            joint_strategy: str = "none", the joint strategy for joint attention, currently only support "front" and "rear"

        Returns:
            * output (Tensor): context output
        """


        sp_world_size = get_sequence_parallel_world_size()  # USP
        sp_rank = get_sequence_parallel_rank()



        joint_tensor_query = torch.chunk(joint_tensor_query, sp_world_size, dim=2)[sp_rank]    # [B, S_text, H, D] -->  [B, S_text, H/ulysses_size, D]
        joint_tensor_key = torch.chunk(joint_tensor_key, sp_world_size, dim=2)[sp_rank]
        joint_tensor_value = torch.chunk(joint_tensor_value, sp_world_size, dim=2)[sp_rank]



        # 3 X (bs, seq_len/N, head_cnt, head_size) -> 3 X (bs, seq_len, head_cnt/N, head_size)
        # scatter 2, gather 1
        if self.use_pack_qkv:
            # (3*bs, seq_len/N, head_cnt, head_size)
            qkv = torch.cat([query, key, value]).contiguous()
            # (3*bs, seq_len, head_cnt/N, head_size)
            qkv = SeqAllToAll4D.apply(
                self.ulysses_pg, qkv, self.scatter_idx, self.gather_idx,
            )
            qkv = torch.chunk(qkv, 3, dim=0)
            query_layer, key_layer, value_layer = qkv

        else:
            # 非打包模式：分别对Q/K/V进行通信拆分
            query_layer = SeqAllToAll4D.apply(
                self.ulysses_pg, query, self.scatter_idx, self.gather_idx ,          # [B, S_image/ulysses_size, H, D] -->   [B, S_image, H/ulysses_size, D]
            )
            key_layer = SeqAllToAll4D.apply(
                self.ulysses_pg, key, self.scatter_idx, self.gather_idx,
            )
            value_layer = SeqAllToAll4D.apply(
                self.ulysses_pg, value, self.scatter_idx, self.gather_idx,
            )

        # Concatenate for joint attention
        # Order: [text, image]
        joint_query = torch.cat([joint_tensor_query, query_layer], dim=1)    # (B, S_txt  + S_img, H/ulysses_size, D_head)
        joint_key = torch.cat([joint_tensor_key, key_layer], dim=1)
        joint_value = torch.cat([joint_tensor_value, value_layer], dim=1)


        out = attention_forward(
            joint_query,
            joint_key,
            joint_value,
            opt_mode="manual",
            op_type="fused_attn_score",
            layout="BNSD"
        )

        if type(out) == tuple:
            context_layer, _, _ = out
        else:
            context_layer = out

        txt_seq_len = joint_tensor_query.shape[1]

        text_out = context_layer[:, :txt_seq_len, :, :].contiguous()  # 强制连续
        image_out = context_layer[:, txt_seq_len:, :, :].contiguous()

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        image_out = SeqAllToAll4D.apply(
            self.ulysses_pg, image_out, self.gather_idx, self.scatter_idx           #  [B,  S_image, H/ulysses_size, D] -->  [B, S_image/ulysses_size, H, D]
        )

        text_out =  get_sp_group().all_gather(text_out, dim=2)      #   (B, S_txt  , H/ulysses_size, D_head) -->   (B, S_txt  , H, D_head)

        output = torch.cat([text_out, image_out], dim=1)
        # out e.g., [s/p::h]
        return output

