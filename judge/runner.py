"""本地裁判 CLI。

三条常用命令：

1. 自检参考实现、跑通用例并打印 golden::

       python3 -m judge.runner verify

2. 把用例导出成文件，便于拿到 CANNLab 上喂给算子::

       python3 -m judge.runner generate --out cases_out

3. 把 NPU 跑出来的结果与 golden 对拍（这一步是整条链路的裁判）::

       python3 -m judge.runner compare --cases cases_out --results results_dir

   或者只对单个结果文件快速比对::

       python3 -m judge.runner compare-one --name all_negative_sim --file y.npy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# 允许 `python3 judge/runner.py` 与 `python3 -m judge.runner` 两种调用方式
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from judge import cases as cases_mod
    from judge import reference as ref
else:
    from . import cases as cases_mod
    from . import reference as ref


# --------------------------------------------------------------- 自检


def cmd_verify(args: argparse.Namespace) -> int:
    """跑一遍全部用例，确认参考实现自洽，并打印 golden 摘要。"""
    all_cases = cases_mod.all_cases()
    print(f"共 {len(all_cases)} 个用例\n")
    header = f"{'name':<24} {'logical (B,M,K)x(B,K,N)':<30} {'dtype':<9} {'y[0:3]':<28} notes"
    print(header)
    print("-" * len(header))

    failed = []
    for c in all_cases:
        try:
            (b, m, k), (_, _, n) = ref.resolve_logical_shape(
                c["x1"], c["x2"], c["transposeX1"], c["transposeX2"]
            )
            y = ref.golden_from_case(c)
        except Exception as exc:  # noqa: BLE001 - 自检就是要暴露问题
            failed.append((c["name"], str(exc)))
            print(f"{c['name']:<24} ERROR: {exc}")
            continue

        shape_str = f"({b},{m},{k})x({b},{k},{n})"
        y_head = np.array2string(y[:3], precision=4, separator=",")
        print(f"{c['name']:<24} {shape_str:<30} {c['dtypeKey']:<9} {y_head:<28} {c['notes'][:40]}")

        if not np.all(np.isfinite(y)):
            failed.append((c["name"], "golden 出现 NaN/Inf"))
        if y.shape != (b,):
            failed.append((c["name"], f"golden shape {y.shape} 应为 ({b},)"))

    print()
    if failed:
        print(f"❌ 自检失败 {len(failed)} 项：")
        for name, why in failed:
            print(f"   - {name}: {why}")
        return 1
    print(f"✅ 自检通过：{len(all_cases)} 个用例的 golden 全部有限且形状正确")
    return 0


# --------------------------------------------------------------- 导出用例


def cmd_generate(args: argparse.Namespace) -> int:
    """把用例与 golden 落盘成 npz，供 CANNLab 侧读取。"""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for c in cases_mod.all_cases():
        if args.name and c["name"] not in args.name:
            continue
        y = ref.golden_from_case(c)
        fp = out_dir / f"{c['name']}.npz"
        np.savez(
            fp,
            x1=c["x1"],
            x2=c["x2"],
            y_golden=y,
            transposeX1=np.array(c["transposeX1"]),
            transposeX2=np.array(c["transposeX2"]),
            dtypeKey=np.array(c["dtypeKey"]),
            notes=np.array(c["notes"]),
        )
        (b, m, k), (_, _, n) = ref.resolve_logical_shape(
            c["x1"], c["x2"], c["transposeX1"], c["transposeX2"]
        )
        index.append(
            {
                "name": c["name"],
                "file": fp.name,
                "B": int(b), "M": int(m), "N": int(n), "K": int(k),
                "x1_storage_shape": list(c["x1"].shape),
                "x2_storage_shape": list(c["x2"].shape),
                "x1_dtype": str(c["x1"].dtype),
                "x2_dtype": str(c["x2"].dtype),
                "transposeX1": bool(c["transposeX1"]),
                "transposeX2": bool(c["transposeX2"]),
                "dtypeKey": c["dtypeKey"],
                "y_golden": [float(v) for v in y],
                "notes": c["notes"],
            }
        )
        print(f"  写出 {fp.name:<28} y={np.array2string(y[:3], precision=4)}")

    (out_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n✅ 共导出 {len(index)} 个用例到 {out_dir}（含 index.json）")
    return 0


# --------------------------------------------------------------- 对拍


def _load_result(results_dir: Path, name: str, dtype: str = "float32") -> np.ndarray:
    """从结果目录里找某个用例的输出。

    依次尝试 ``<name>.npy`` / ``<name>.bin``。
    """
    npy = results_dir / f"{name}.npy"
    if npy.exists():
        return np.load(npy, allow_pickle=False)
    binp = results_dir / f"{name}.bin"
    if binp.exists():
        return np.fromfile(binp, dtype=np.dtype(dtype))
    raise FileNotFoundError(f"在 {results_dir} 下找不到 {name}.npy 或 {name}.bin")


def cmd_compare(args: argparse.Namespace) -> int:
    """把 NPU 结果与 golden 对拍，给出每个用例的精度结论。"""
    cases_dir = Path(args.cases)
    results_dir = Path(args.results)
    index = json.loads((cases_dir / "index.json").read_text(encoding="utf-8"))

    if args.name:
        index = [it for it in index if it["name"] in args.name]
    if not index:
        print("❌ 没有匹配的用例")
        return 1

    print(f"对拍 {len(index)} 个用例：results={results_dir}\n")
    n_pass = 0
    failures = []
    for it in index:
        name = it["name"]
        try:
            actual = _load_result(results_dir, name)
        except FileNotFoundError as exc:
            print(f"[MISS] {name:<24} {exc}")
            failures.append(name)
            continue

        golden = np.asarray(it["y_golden"], dtype=np.float32)
        res = ref.compare(actual, golden, dtype_key=it["dtypeKey"])
        print(f"{name:<24} {res.summary()}")
        if res.detail:
            print("    " + res.detail.replace("\n", "\n    "))
        if res.passed:
            n_pass += 1
        else:
            failures.append(name)

    print()
    print(f"结果：{n_pass}/{len(index)} 通过")
    if failures:
        print(f"❌ 未通过：{', '.join(failures)}")
        print("\n提示：先看是不是所有非对齐/全负用例都挂——那通常是尾块或 MaxSim 初值问题；")
        print("      如果只有 k8192_max / large_square 挂，优先怀疑 FP32 累加精度或 tiling 切分。")
        return 1
    print("✅ 全部通过")
    return 0


def cmd_compare_one(args: argparse.Namespace) -> int:
    """对单个结果文件快速比对。"""
    c = cases_mod.find_case(args.name)
    golden = ref.golden_from_case(c)
    actual = np.load(args.file, allow_pickle=False) if args.file.endswith(".npy") else np.fromfile(args.file, dtype=np.float32)
    res = ref.compare(actual, golden, dtype_key=args.dtype or c["dtypeKey"])
    print(res.summary())
    if res.detail:
        print(res.detail)
    print(f"golden = {golden[:8]}")
    print(f"actual = {np.asarray(actual)[:8]}")
    return 0 if res.passed else 1


# --------------------------------------------------------------- 得分估算


def cmd_score(args: argparse.Namespace) -> int:
    """按赛题公式把"相对基线的加速比"换算成得分，用于估算排名。"""
    base = args.base
    print(f"基线（拆分实现总耗时 T） = {base} us")
    print(f"共 {ref.NUM_TEST_POINTS} 个测试点，单点得分 = 100 / (1 + log_1.5(t/T))\n")
    print(f"{'加速比':<10} {'t (us)':<12} {'单点得分':<10}")
    print("-" * 34)
    for speedup in [0.5, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0]:
        t = base / speedup
        print(f"{speedup:<10.2f} {t:<12.2f} {ref.case_score(t, base):<10.2f}")
    if args.time is not None:
        print(f"\n当前提交 t={args.time} us -> 加速比 {base / args.time:.3f}x, 单点得分 {ref.case_score(args.time, base):.2f}")
    return 0


# --------------------------------------------------------------- 入口


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="judge.runner",
        description="BatchMatmulMaxSum 本地裁判：自检 / 导出用例 / 对拍 NPU 结果 / 估算得分",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("verify", help="自检参考实现并打印全部用例 golden")
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("generate", help="导出用例与 golden 到目录")
    sp.add_argument("--out", default="cases_out", help="输出目录（默认 cases_out）")
    sp.add_argument("--name", nargs="*", help="只导出指定用例名")
    sp.set_defaults(func=cmd_generate)

    sp = sub.add_parser("compare", help="对拍 NPU 结果目录")
    sp.add_argument("--cases", default="cases_out", help="generate 出来的用例目录")
    sp.add_argument("--results", required=True, help="存放算子输出的目录（<name>.npy 或 <name>.bin）")
    sp.add_argument("--name", nargs="*", help="只对拍指定用例名")
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("compare-one", help="对拍单个结果文件")
    sp.add_argument("--name", required=True, help="用例名")
    sp.add_argument("--file", required=True, help="结果文件 .npy 或 .bin(float32)")
    sp.add_argument("--dtype", help="覆盖 dtypeKey")
    sp.set_defaults(func=cmd_compare_one)

    sp = sub.add_parser("score", help="按赛题公式估算得分")
    sp.add_argument("--base", type=float, required=True, help="拆分实现基线总耗时 T (us)")
    sp.add_argument("--time", type=float, help="当前提交耗时 t (us)")
    sp.set_defaults(func=cmd_score)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
