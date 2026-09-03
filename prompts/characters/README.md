# 角色 Prompt 库

每个角色目录包含稳定的角色 skill `base.md`、独立的 Prompt Evolution `manifest.json` 和版本目录。controller 根据公开状态中的 `class` 只加载当前角色目录。版本文件直接保存 `role_skill` 与 `scene_skills`；`manifest.json` 的 `character_prompt_version` 用于追踪角色基础库变更。`config.prompt_evolution.policy_dir` 应指向本目录。
