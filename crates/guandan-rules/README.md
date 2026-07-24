# guandan-rules

`guandan-rules` 是独立的四人两副牌掼蛋规则引擎，规则基线来自东南大学发布的
[《扑克牌（掼蛋）比赛规则（两副牌）》](https://xxgk.seu.edu.cn/_upload/article/files/44/c8/f455e1d04d2a998e40454931740a/4f853bb4-29b9-45dc-9c56-7627ed4c9726.pdf)。

它不依赖 `ddz-core`，也不把四人、花色、两副实体牌或升级逻辑塞进三人斗地主
领域模型。Web 通过独立的 `guandan-pyo3` 和 `/api/guandan/games` 会话边界调用
本包，不在 Python 或 React 中复制规则。

## 已实现规则

- 108 张实体牌、四人对家组队、每人 27 张及可复现发牌。
- 当前级牌、两张红桃级牌“逢人配”，且不能替代大小王。
- 单张、对子、三张、三带二、五张顺子、三连对、两连三张（钢板）。
- 四至十张炸弹、五张同花顺和四王，以及 PDF 规定的完整优先级。
- 牌型相同且张数相同时的比较；三带二按三张、连续组合按最高组合比较。
- 事务式出牌、过牌、一墩结束、出完后的对家接风和四人名次。
- 双下升三级、头游对家第三升两级、头游对家末游升一级，打 A 获胜结束。
- 单下/双下进贡、红桃级牌排除、还贡不超过 10、两张大王抗贡及后续出牌权。
- 剩余六张主动报牌、剩余十张有问必报的查询策略。

比赛组织条款（洗牌次数、迟到、违规扣分、非法信息和限时）属于裁判流程，
不会混入确定性的牌局状态机。

## 模块边界

```text
card/       实体牌、牌面、手牌、座位与搭档
movement/   牌型识别、连续序列和比较
game/       发牌、回合状态机、升级、进贡
movement/generate/  面向 Web 与 AI 的合法牌生成
config.rs   可序列化规则配置
report.rs   报牌义务查询
```

所有状态变更先在克隆状态上验证，失败不会留下部分出牌或部分换牌。源文件按单一
职责拆分，规则强制走公开领域类型，不向上层暴露裸数组编码协议。

## 使用

```rust
use guandan_rules::{Action, Rank, Round, Seat};

let mut round = Round::new(42, Rank::Two, Seat::ZERO)?;
let current = round.current_player();
let first_card = round.hand(current).cards().next().unwrap();
round.step(current, Action::Play(vec![first_card]))?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

配置文件位于 `configs/rules/guandan_two_deck.yaml`。

## 验证

```powershell
cargo test -p guandan-rules
```
