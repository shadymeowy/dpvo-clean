# trt_bench.py
import os
import time
import numpy as np
import torch
import tensorrt as trt
import onnx
import cupy as cp

import ctypes

# Make sure the TensorRT plugins library is loaded & registered
ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)
trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.WARNING), "")

IN_NAMES = ["net", "inp", "corr", "nix", "njx", "ukk", "ujk"]
OUT_NAMES = ["net_out", "d", "w"]


def _torch_dtype_from_trt(dt: trt.DataType):
    return {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.BOOL: torch.bool,
    }[dt]


def _match_trt_dtype(t: torch.Tensor, dt: trt.DataType):
    want = _torch_dtype_from_trt(dt)
    if t.dtype == want:
        return t
    # Only cast float tensors; casting index tensors changes values mod 2^32
    if t.is_floating_point():
        return t.to(want)
    # For indices, prefer to cast BEFORE building the engine / exporting ONNX
    return t


def _check_no_unique(onnx_path: str):
    m = onnx.load(onnx_path)
    if any(n.op_type == "Unique" for n in m.graph.node):
        raise RuntimeError(
            "ONNX contains op 'Unique' (TRT unsupported). Remove torch.unique from the graph."
        )


def build_engine_from_example(
    onnx_path: str,
    plan_path: str,
    example_inputs: dict,
    workspace_mb: int = 4096,
    fp16: bool = False,
):
    logger = trt.Logger(trt.Logger.WARNING)

    # Sanity: no Unique
    m = onnx.load(onnx_path)
    if any(n.op_type == "Unique" for n in m.graph.node):
        raise RuntimeError("ONNX contains op 'Unique' (TRT unsupported).")

    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    builder = trt.Builder(logger)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("[TRT][Parser]", parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX.")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Profile with EXACT example shapes (min=opt=max) for a guaranteed build
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        t = network.get_input(i)
        shp = tuple(example_inputs[t.name].shape)
        profile.set_shape(t.name, shp, shp, shp)
    config.add_optimization_profile(profile)

    # TRT 10+: build serialized blob, then deserialize to engine
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Failed to build serialized network.")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("Failed to deserialize engine from serialized blob.")

    with open(plan_path, "wb") as f:
        f.write(serialized)
    print(f"[TRT] Engine saved -> {plan_path}")
    return plan_path


def build_engine_dynamic_fp16(
    onnx_path: str,
    plan_path: str,
    # set a good N range for your use case
    minN: int = 1,
    optN: int = 2280,
    maxN: int = 8192,
    workspace_mb: int = 8192,  # give TRT plenty of room
    enable_fp16: bool = True,
    prefer_int32_indices: bool = True,
):
    logger = trt.Logger(trt.Logger.WARNING)

    # Guard: no ONNX Unique
    m = onnx.load(onnx_path)
    if any(n.op_type == "Unique" for n in m.graph.node):
        raise RuntimeError("ONNX contains op 'Unique' (TRT unsupported).")

    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    builder = trt.Builder(logger)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("[TRT][Parser]", parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX.")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)

    # Maximize optimizer effort (if available on your TRT build)
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = 5  # 0..5, higher = more search
    if hasattr(config, "set_flag"):
        # FP16 kernels
        if enable_fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        # Structured sparsity (if your linear weights are 2:4 sparse)
        if hasattr(trt.BuilderFlag, "SPARSE_WEIGHTS"):
            config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
        # Honor per-tensor precisions if you set any (default: not needed)
        if hasattr(config, "set_precision_constraints"):
            config.set_precision_constraints(trt.PrecisionConstraints.OBEY)

    # Use all tactic sources (if the API exists)
    if hasattr(config, "set_tactic_sources"):
        try:
            config.set_tactic_sources(
                trt.TacticSource.CUBLAS_LT
                | trt.TacticSource.CUBLAS
                | trt.TacticSource.CUDNN
                | trt.TacticSource.JIT_CONVOLUTIONS
            )
        except Exception:
            pass

    # Dynamic profile on N (axis 1 for [1,N,F], axis 0 for [N])
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        t = network.get_input(i)
        name, shp = t.name, list(t.shape)
        if len(shp) == 3:  # [1, N, F]
            F = shp[2]
            profile.set_shape(
                name, min=(1, minN, F), opt=(1, optN, F), max=(1, maxN, F)
            )
        elif len(shp) == 1:  # [N]
            profile.set_shape(name, min=(minN,), opt=(optN,), max=(maxN,))
        else:
            raise ValueError(f"Unexpected rank for input '{name}': {shp}")
    config.add_optimization_profile(profile)

    # Build serialized blob, then deserialize
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Failed to build serialized network.")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("Failed to deserialize engine.")

    with open(plan_path, "wb") as f:
        f.write(serialized)
    print(f"[TRT] FP16 plan saved -> {plan_path}")
    return engine


def load_engine(plan_path: str):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(plan_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("Failed to deserialize engine.")
    context = engine.create_execution_context()
    return engine, context


def _bind_io(engine, context, inputs_torch: dict):
    """
    Set input shapes, bind device pointers for inputs & outputs.
    Returns (outputs_torch: dict, stream: cupy Stream).
    """
    stream = cp.cuda.Stream(non_blocking=True)

    # Bind inputs
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        if is_input:
            t = inputs_torch[name].contiguous()
            # cast dtype to engine expectation if needed
            t = _match_trt_dtype(t, engine.get_tensor_dtype(name))
            inputs_torch[name] = t  # keep reference
            context.set_input_shape(name, tuple(t.shape))
            context.set_tensor_address(name, t.data_ptr())

    # Allocate & bind outputs
    outputs = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            out_shape = context.get_tensor_shape(name)
            out_dtype = _torch_dtype_from_trt(engine.get_tensor_dtype(name))
            out_t = torch.empty(tuple(out_shape), dtype=out_dtype, device="cuda")
            context.set_tensor_address(name, out_t.data_ptr())
            outputs[name] = out_t

    return outputs, stream


def trt_run_once(engine, context, inputs_torch: dict, warmup: int = 0):
    outs, stream = _bind_io(engine, context, inputs_torch)
    # warmup
    for _ in range(warmup):
        context.execute_async_v3(stream.ptr)
    stream.synchronize()
    # run once
    context.execute_async_v3(stream.ptr)
    stream.synchronize()
    return {k: v.detach().cpu().numpy() for k, v in outs.items()}


def trt_benchmark_same_inputs(
    update_module,
    dct,
    onnx_path="update_trt.onnx",
    plan_path="update_trt_fp16.plan",
    fp16=False,
    workspace_mb=4096,
    n_warm=20,
    n_iter=100,
):
    # Build engine (exact shapes from your tensors)
    example_inputs = {
        "net": dct["net_in"],
        "inp": dct["inp"],
        "corr": dct["corr"],
        "nix": dct["nix"],
        "njx": dct["njx"],
        "ukk": dct["ukk"],
        "ujk": dct["ujk"],
    }
    if False:  # not os.path.exists(plan_path):
        # build_engine_from_example(
        #     onnx_path, plan_path, example_inputs, workspace_mb=workspace_mb, fp16=fp16
        # )
        engine = build_engine_dynamic_fp16(
            onnx_path="update_trt.onnx",
            plan_path="update_trt_fp16.plan",
            minN=1,
            optN=dct["net_in"].shape[1],
            maxN=8192,
            workspace_mb=8192,
            enable_fp16=True,
        )

    engine, context = load_engine("update_trt_fp16.plan")

    # Torch reference (same inputs)
    update_module.eval()
    with torch.inference_mode():
        ref_net, ref_d, ref_w = update_module(
            dct["net_in"],
            dct["inp"],
            dct["corr"],
            dct["nix"],
            dct["njx"],
            dct["ukk"],
            dct["ujk"],
        )
    ref_net_np = ref_net.detach().cpu().numpy()
    ref_d_np = ref_d.detach().cpu().numpy()
    ref_w_np = ref_w.detach().cpu().numpy()

    # Prepare TRT input dict (same CUDA tensors)
    trt_inputs = {
        "net": dct["net_in"],
        "inp": dct["inp"],
        "corr": dct["corr"],
        "nix": dct["nix"],
        "njx": dct["njx"],
        "ukk": dct["ukk"],
        "ujk": dct["ujk"],
    }

    # (Optional) force indices to int32 if engine expects INT32
    for name in ["nix", "njx", "ukk", "ujk"]:
        want = _torch_dtype_from_trt(engine.get_tensor_dtype(name))
        if trt_inputs[name].dtype != want:
            trt_inputs[name] = trt_inputs[name].to(want)

    # Warmup & correctness
    _ = trt_run_once(engine, context, trt_inputs, warmup=n_warm)
    out = trt_run_once(engine, context, trt_inputs, warmup=0)
    trt_net, trt_d, trt_w = out["net_out"], out["d"], out["w"]

    def stats(name, a, b):
        mae = np.max(np.abs(a - b))
        mx = np.max(np.abs(a))
        print(f"{name}: max|Δ|={mae:.6g}   max|ref|={mx:.6g}")

    stats("net_out", ref_net_np, trt_net)
    stats("d", ref_d_np, trt_d)
    stats("w", ref_w_np, trt_w)

    # ---- GPU timing with CuPy events on our TRT stream ----
    outs, stream = _bind_io(engine, context, trt_inputs)
    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record(stream)
    for _ in range(n_iter):
        context.execute_async_v3(stream.ptr)
    end.record(stream)
    end.synchronize()
    ms = cp.cuda.get_elapsed_time(start, end) / n_iter
    print(f"TensorRT (GPU): {ms:.3f} ms/iter")

    # Host-latency (inc. enqueue + sync)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        context.execute_async_v3(stream.ptr)
        stream.synchronize()
    t1 = time.perf_counter()
    print(f"TensorRT (latency): {(t1 - t0) * 1000 / n_iter:.3f} ms/iter")

    return ms
