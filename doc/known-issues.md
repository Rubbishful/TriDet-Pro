# Known Issues & Workarounds

## Scheduler 初始化警告（已屏蔽）

**现象**：
```
UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`.
```

**根因**：PyTorch >= 2.1 中，`_LRScheduler.__init__` 会调用 `_initial_step()` → `self.step()` 来初始化 LR 状态。此时 `optimizer._step_count` 仍为 0（optimizer 尚未执行任何 step），触发 PyTorch 的 protective warning。

这是 PyTorch 框架级别的设计行为，不是代码顺序错误。实际训练循环中 `optimizer.step()` 始终在 `scheduler.step()` 之前执行，仅在 scheduler 构造阶段有此一次性警告。

**处理方式**：`libs/utils/train_utils.py:make_scheduler()` 中用 `warnings.catch_warnings()` 在构造时屏蔽该警告。不影响训练正确性。

**相关文件**：[libs/utils/train_utils.py:137-142](../libs/utils/train_utils.py#L137-L142)

---

## AMP 已实现（2026-07-04）

混合精度训练已实现，使用 `torch.amp.GradScaler('cuda')` + `torch.amp.autocast('cuda')`。

- 默认关闭，通过 `--amp` CLI flag 或配置 `train_cfg.use_amp: True` 启用
- IoU 迭代训练配置 `configs/thumos_i3d_iou_tal_iteration.yaml` 中已启用
- 其他配置文件未修改，保持 float32 训练

**相关文件**：
- [libs/utils/train_utils.py:298-327](../libs/utils/train_utils.py#L298-L327) — AMP 训练逻辑
- [train.py:243](../train.py#L243) — `--amp` flag
- [tools/search_hyperparams.py:635](../tools/search_hyperparams.py#L635) — `--amp` flag

---

## DataLoader pin_memory 已启用

`libs/datasets/datasets.py:make_data_loader()` 已添加 `pin_memory=True` + `prefetch_factor=2`，所有训练自动受益。

**相关文件**：[libs/datasets/datasets.py:23-35](../libs/datasets/datasets.py#L23-L35)

---

## Grid Search 布尔值解析修复

`tools/search_hyperparams.py:generate_trials()` 中 `choice` 类型参数的 grid 解析已支持 `True`/`False` 布尔值。

**相关文件**：[tools/search_hyperparams.py:306-316](../tools/search_hyperparams.py#L306-L316)

---

## PyTorch 版本

实际环境使用 **PyTorch 2.6.0+cu124**（README 中标注的 2.0.1 已过时）。`torch.cuda.amp` → `torch.amp` 的 API 迁移已基于 2.6 完成。
