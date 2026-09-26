"""Explicit UUID / PCI / EGL binding; never fall back to an unapproved GPU."""

import ctypes
import ctypes.util
import json
import os
import subprocess
import sys
import time


def pci_key(bus):
    # nvidia-smi uses an eight-digit PCI domain, sysfs normally uses four digits.
    domain, bus, device = bus.lower().split(":")
    return f"{int(domain, 16):04x}:{bus}:{device}"


def gpu_model_matches(name, model):
    patterns = {"4090": "RTX 4090", "5880": "RTX 5880", "pro5000": "RTX PRO 5000"}
    if model not in patterns:
        raise ValueError(f"Unsupported GPU model: {model}")
    return patterns[model] in name


def gpu_inventory(model="4090"):
    gpu_model_matches("", model)
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id", "--format=csv,noheader"], text=True
    )
    cards = []
    for line in output.splitlines():
        index, name, uuid, bus = [x.strip() for x in line.split(",")]
        if gpu_model_matches(name, model):
            cards.append({"index": int(index), "name": name, "uuid": uuid, "pci": pci_key(bus)})
    if len(cards) != 2:
        raise RuntimeError(f"Expected exactly two {model}s, found {cards}; refusing automatic fallback")
    return cards


def egl_inventory():
    """把 EGL 设备映射到 PCI 地址；不能假设 EGL、CUDA、nvidia-smi 编号相同。

    这里只枚举设备，不创建渲染 context，也不在任何 GPU 上运行模型。
    只接受具有 EGL_NV_device_cuda 的 NVIDIA EGL 设备；Mesa 可能为同一 PCI
    显卡再暴露一个无法正常初始化的 DRM 条目，不能拿它覆盖 NVIDIA 条目。
    使用 CUDA device 属性，通过 Driver API 查询 PCI 地址。
    无法验证映射时直接报错，绝不退回未经选择的默认 GPU 0。
    """
    egl = ctypes.CDLL(ctypes.util.find_library("EGL") or "libEGL.so.1")
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    egl.eglGetProcAddress.restype = ctypes.c_void_p

    def proc(name, restype, *args):
        address = egl.eglGetProcAddress(name.encode())
        if not address:
            raise RuntimeError(f"Missing EGL function {name}")
        return ctypes.CFUNCTYPE(restype, *args)(address)

    query = proc(
        "eglQueryDevicesEXT",
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
    )
    string = proc("eglQueryDeviceStringEXT", ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)
    attrib = proc(
        "eglQueryDeviceAttribEXT",
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_ssize_t),
    )
    devices = (ctypes.c_void_p * 64)()
    count = ctypes.c_int()
    if not query(64, devices, ctypes.byref(count)):
        raise RuntimeError("EGL device enumeration failed")
    found = {}
    for i in range(count.value):
        extensions = string(devices[i], 0x3055) or b""  # EGL_EXTENSIONS
        if b"EGL_NV_device_cuda" not in extensions.split():
            continue
        ordinal = ctypes.c_ssize_t()
        if attrib(devices[i], 0x323A, ctypes.byref(ordinal)):  # EGL_CUDA_DEVICE_NV
            cuda = ctypes.CDLL("libcuda.so.1")
            if cuda.cuInit(0) == 0:
                buffer = ctypes.create_string_buffer(64)
                if cuda.cuDeviceGetPCIBusId(buffer, 64, int(ordinal.value)) == 0:
                    found[pci_key(buffer.value.decode())] = i
    return found


def cards_and_envs(model="4090"):
    cards = gpu_inventory(model)
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    mapping = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import json; from roboscope.runtime.gpu import egl_inventory; print(json.dumps(egl_inventory()))",
            ],
            env=env,
            text=True,
        )
    )
    envs = []
    for card in cards:
        if card["pci"] not in mapping:
            raise RuntimeError(f"Cannot map {model} EGL")
        card["egl"] = mapping[card["pci"]]
        worker = os.environ.copy()
        worker.update(
            CUDA_VISIBLE_DEVICES=card["uuid"],
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            MUJOCO_EGL_DEVICE_ID=str(card["egl"]),
            MUJOCO_GL="egl",
            PYOPENGL_PLATFORM="egl",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1",
        )
        envs.append(worker)
    return cards, envs


def run_workers(commands, envs, logs):
    """任意 worker 失败则停止同轮其他 worker；Ctrl-C 清理子进程组，不留后台评测。"""
    processes = []
    handles = []
    try:
        for cmd, env, log in zip(commands, envs, logs, strict=True):
            f = log.open("a")
            handles.append(f)
            processes.append(
                subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            )
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError("Worker failed; inspect " + str(logs))
            time.sleep(1)
        if any(p.returncode for p in processes):
            raise RuntimeError("Worker failed; inspect " + str(logs))
    finally:
        import signal

        for p in processes:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        for f in handles:
            f.close()
