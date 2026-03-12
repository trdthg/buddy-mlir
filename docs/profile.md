# Qwen3 K1 Perf

程序先由你自己启动。

进入 `prefill` 阶段后，在另一个终端里直接复制下面这一整段。它会用 `ps` 找 `buddy-qwen3-0.6b-run` 的 PID，顺序跑完 `perf stat`、`perf record`、`perf report`，结果统一输出到一个时间戳目录里。

```bash
cd ~/repo/cross-compile/buddy-qwen3-0.6b-rvv-package

export OUT=perf_$(date +%Y%m%d_%H%M%S)
export WINDOW=10
mkdir -p "$OUT"

export PID=$(ps -eo pid=,args= | awk '/[b]uddy-qwen3-0\.6b-run/ {print $1; exit}')
if [ -z "$PID" ]; then
  echo "buddy-qwen3-0.6b-run 没在运行"
  exit 1
fi

echo "PID=$PID"
echo "OUT=$OUT"

sudo perf stat -p "$PID" -o "$OUT/01_overview.txt" \
  -e cycles,instructions,stalled-cycles-frontend,stalled-cycles-backend,branches,branch-misses \
  sleep "$WINDOW"

sudo perf stat -p "$PID" -o "$OUT/02_mem_tlb.txt" \
  -e l2_load_access,l2_load_miss,l2_ar_channel_stall_cycle,l2_aw_channel_stall_cycle,dTLB-loads,dTLB-load-misses,dTLB-stores,dTLB-store-misses,jtlb_miss \
  sleep "$WINDOW"

sudo perf stat -p "$PID" -o "$OUT/03_lsu_vpu.txt" \
  -e lsu_load_commit_stall,lsu_load_raw_stall,lsu_store_commit_stall,rf_raw_stall,eu_stall,vpu_stall_pipe0,vpu_stall_pipe1,vpu_vlsu_stall_pipe0,vpu_vlsu_stall_pipe1,vidu_vec0_depend_stall,vidu_vec1_depend_stall \
  sleep "$WINDOW"

sudo perf record -F 199 -e cpu-clock -g -p "$PID" -o "$OUT/prefill.data" -- sleep "$WINDOW"

sudo perf report -i "$OUT/prefill.data" --stdio --no-children --percent-limit 0.5 > "$OUT/04_hotspots_all.txt"

grep -E 'subgraph0_prefill|subgraph0_decode|expf|malloc|free|__kmp' "$OUT/04_hotspots_all.txt" > "$OUT/05_hotspots_focus.txt" || true

echo
echo "done: $OUT"
ls -lh "$OUT"
```

结果重点看这几个文件：

- `01_overview.txt`
- `02_mem_tlb.txt`
- `03_lsu_vpu.txt`
- `04_hotspots_all.txt`
- `05_hotspots_focus.txt`
