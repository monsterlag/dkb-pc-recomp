# NVN 2 compute overlap

`runtime/nvn_api_nvn2.c`, `nvnCommandBufferSetComputeOverlapMode` preserves the replay's existing serialized
execution. This call controls hardware compute/graphics scheduling; it does not change shader bindings, resource
contents, or which dispatches execute. Allowing concurrent execution does not require a host to overlap the work.

The Bananza base SDK establishes the contract without relying on the function's name:

- Proc-table entry 185, SDK offset `0x5e9780`, takes the command buffer in `x0` and a 32-bit mode in `w1`, returns
  void, and appends an eight-byte queue control record with opcode `0x27` and the mode.
- The queue control reader at SDK offset `0x5c5784` sends that record to the scheduling helper at `0x5e96c0`.
  When the queue supports this control, changing the mode writes compute method byte offset `0x1e8` inside an
  atomic sequence. Mode zero sets bit zero; a nonzero mode clears it.
- NVIDIA's public Ampere compute definition names that method `SET_SM_SCG_CONTROL` and bit zero
  `COMPUTE_IN_GRAPHICS`. The call selects where compute is scheduled; the SDK reader changes no shader or resource
  state. See [NVIDIA, Ampere compute class](https://github.com/NVIDIA/open-gpu-doc/blob/master/classes/compute/clc7c0.h).

The replay provides a conservative implementation of either choice. `runtime/vk_exec_draw.c`, `op_dispatch` and
`op_dispatch_ind`, call `vk_exec_barrier` before compute work. `runtime/vk_exec.c`, `vk_exec_pass_begin`, calls
`barrier_to` before graphics work. Those full barriers order graphics and compute in both directions on the single
host queue. No command-stream state needs to be added for this scheduling preference.

A 60-second tutorial run observed mode `1`, called from guest LR `0x70002d27f8`. That selects the SDK's separate
compute scheduling path; the host retains its existing barriers and serialized execution. The implementation gives
up potential overlap performance. The observation covers this call in the tutorial, with full build and live
validation of the explicit handler still required; it does not establish behavior for every game scene or SDK.

This evidence applies only to compute overlap. `SetGeometryProcessingMode` changes mesh-pipeline state, and
`SetRecordingMode` also selects engine-specific synchronization paths in the SDK. They remain separately tracked
in issue 6.
