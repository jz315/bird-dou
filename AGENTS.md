# BIRD-Dou 代理协作指南

## 项目边界

- `crates/ddz-core`：纯领域模型、状态编码和可跨层复用的不变量；不得依赖规则、批处理、Python 或 Web。
- `crates/ddz-rules`：权威斗地主规则、状态迁移、可见性和结算；只依赖 `ddz-core`。
- `crates/ddz-batch`：多环境所有权、缓存、事务和 packed 协议；规则逻辑必须委托给 `ddz-rules`。
- `crates/ddz-pyo3`：Python 边界；`python/` 为训练、评估、命令行和 Web 服务层；`web/` 是 React/Vite 前端。
- `crates/guandan-rules`：独立的四人两副牌掼蛋领域与规则状态机；不得依赖或污染固定三人、54 张牌的 `ddz-core`。
- `crates/guandan-pyo3`：掼蛋专用 Python 边界；保持与尚在迁移的 `ddz-pyo3`、`ddz-search` 解耦。
- 保持依赖方向：`ddz-core -> ddz-rules -> ddz-batch -> ddz-pyo3/python/web`。不要在上层以外复制规则实现。

## Rust 约定

- Workspace 使用 Rust 2021，最低 Rust 1.85，并禁止 `unsafe`。
- 领域值优先使用 `Seat`、`SeatMap`、`RankCounts` 等强类型；不要以裸整数或数组绕开已有验证。
- `Game::step` 和批量操作须保持事务语义：失败不得留下部分状态或缓存更新。
- 观察必须保持信息集安全：不得泄露种子、牌序、私有发牌计划或未公开的加倍选择。
- 当前规则/批处理重构是破坏性迁移。除非任务明确要求，不要恢复 `V1`/`V2`、`PostBidGame` 等旧接口来迁就 `ddz-search` 或 `ddz-pyo3`。

## 验证

按改动范围运行最小相关检查：

```powershell
cargo test -p ddz-core
cargo test -p ddz-rules
cargo test -p ddz-batch
cargo test -p guandan-rules
cargo test -p guandan-pyo3
```

- 修改 Rust 且工作区兼容时，再运行 `cargo test --workspace`；当前 `ddz-search` 与 `ddz-pyo3` 仍在独立迁移中，不要把它们的旧 API 编译错误归因于 core/rules/batch 改动。
- 修改 Python：使用 `.venv\Scripts\python.exe -m pytest` 运行相关测试；按 `pyproject.toml` 运行 Ruff 和 Mypy。
- 修改前端：在 `web/` 下运行 `npm run typecheck`；需要交付静态页面时运行 `npm run build`。

## Web 本地运行

```powershell
.\scripts\build_guandan_native.ps1
cd web
npm run build
cd ..
.\.venv\Scripts\python.exe -m birddou.web.server --open

```

服务默认监听 `http://127.0.0.1:8765`，前端开发代理指向同一端口。

## 工作区卫生

- 不提交或手工编辑 `target/`、`web/dist/`、`node_modules/`、虚拟环境、缓存和生成的 research artifacts。
- 保留用户已有的未提交改动；不要用 reset、checkout 或大范围删除来清理工作区。
- 新增/变更公开协议、配置或状态编码时，补充相邻测试，并同步相应 README 或协议文档。
