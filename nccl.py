parallel_state.py 수정
import triton
import triton.language as tl

# ── sentinel: x의 첫 원소를 읽고 그대로 쓴다 (값 보존, mutation 선언) ──────
@triton.jit
def _sentinel_ag_begin_kernel(ptr, BLOCK: tl.constexpr):
    # ptr[0]을 읽어서 그대로 다시 씀 → 값 변경 없음, but mutation으로 인식
    val = tl.load(ptr)
    tl.store(ptr, val)

@triton.jit
def _sentinel_ag_end_kernel(ptr, BLOCK: tl.constexpr):
    val = tl.load(ptr)
    tl.store(ptr, val)

def warmup_sentinel_kernels(device: torch.device) -> None:
    buf = torch.zeros(1, dtype=torch.float32, device=device)
    _sentinel_ag_begin_kernel[(1,)](buf, BLOCK=1)
    _sentinel_ag_end_kernel[(1,)](buf, BLOCK=1)
    torch.cuda.synchronize()

# ── in-place mutation op: x[0]에 write → inductor가 DCE 불가 ────────────────
def _op_sentinel_ag_begin(x: torch.Tensor) -> torch.Tensor:
    # x의 첫 원소 포인터에 커널 실행 (값은 유지됨)
    _sentinel_ag_begin_kernel[(1,)](x.view(-1)[:1].contiguous(), BLOCK=1)
    return x

def _op_sentinel_ag_end(x: torch.Tensor) -> torch.Tensor:
    _sentinel_ag_end_kernel[(1,)](x.view(-1)[:1].contiguous(), BLOCK=1)
    return x

# ── fake_impl: x를 mutate하는 것처럼 선언 → inductor DCE 방지 핵심 ──────────
def _sentinel_fake(x: torch.Tensor) -> torch.Tensor:
    # clone()으로 반환 → inductor가 "이 op는 새 값을 만든다"고 인식
    return x.clone()

direct_register_custom_op(
    op_name="sentinel_ag_begin",
    op_func=_op_sentinel_ag_begin,
    fake_impl=_sentinel_fake,
)
direct_register_custom_op(
    op_name="sentinel_ag_end",
    op_func=_op_sentinel_ag_end,
    fake_impl=_sentinel_fake,
)
sequence_parallelism.py — _insert_ag_sentinels 수정
fake_impl이 x.clone()을 반환하므로 end_node의 출력이 node(all_gather)와 다른 tensor로 인식됩니다. 하위 노드 연결을 명확히 해야 합니다.
def _insert_ag_sentinels(self, graph: fx.Graph) -> None:
    found = 0
    for node in list(graph.nodes):
        if not (
            node.op == "call_function"
            and node.target == torch.ops.vllm.all_gather.default
        ):
            continue

        found += 1

        # ① all_gather의 원래 input
        tensor_arg = node.args[0]

        # ② AG_BEGIN: all_gather 직전 삽입
        with graph.inserting_before(node):
            begin_node = graph.call_function(
                torch.ops.vllm.sentinel_ag_begin.default,
                args=(tensor_arg,),
            )
        # all_gather의 입력을 begin_node 출력으로 교체
        node.args = (begin_node,) + node.args[1:]

        # ③ AG_END: all_gather 직후 삽입
        # 먼저 node의 모든 user를 수집 (inserting_after 전에)
        original_users = list(node.users.keys())

        with graph.inserting_after(node):
            end_node = graph.call_function(
                torch.ops.vllm.sentinel_ag_end.default,
                args=(node,),
            )

        # ④ all_gather의 하위 사용자를 end_node로 교체
        #    (end_node 자신은 제외)
        for user in original_users:
            user.replace_input_with(node, end_node)

    logger.warning("[SENTINEL] inserted around %d all_gather nodes", found)