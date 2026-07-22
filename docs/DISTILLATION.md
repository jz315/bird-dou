# Privileged Teacher and Information-Set Distillation

## M6 full-state Teacher

`PrivilegedTeacher` is a training-only network. It receives the same public
`RaggedBatch` as the Student plus one validated allocation of the unknown pool to
the next relative player; the previous player's hand is the exact remainder.
Together with the observer's public own hand, these form `[B, 3, 15]` remaining
hand counts.

Rank, player, and count embeddings create 45 full-hand tokens. A two-layer global
Transformer mixes them and three player pools fuse with the public E020 state.
The existing role adapter, action encoder, and multi-task heads then score the
same canonical legal-action rows. `PrivilegedCritic` exposes the resulting MC-Q
for centralized terminal-return regression.

The Student model does not own or reference this module. Its state dictionary is
tested to contain no Teacher, Oracle, full-hand, privileged, or true-assignment
keys. For a fixed public information set, changing only an external true hidden
label cannot change Student output, while it changes the Teacher full state.

## Oracle Dropout

Teacher forward accepts an explicit Oracle Dropout probability in `[0,1]`.
The observer's known hand is always retained; each hidden player/rank count is
independently replaced by a learned mask embedding. At probability 1, Teacher
output is required to be bit-identical for any two capacity-valid hidden
allocations of the same public state. Training schedules can therefore move from
full information through partial masking without changing the Student boundary.

## Information-set-consistent KD

Directly distilling `Teacher(true_state)` gives mutually incompatible targets for
hidden states sharing one information set. IS-KD instead:

1. runs the Student Belief CRF;
2. draws `K` exact capacity-preserving hidden hands;
3. optionally includes the training example's true hand;
4. scores every legal action with the frozen Teacher for every hand;
5. averages Q across hidden states;
6. constructs one segment-softmax target from that average.

```text
Q_bar(I,a) = mean_k Q_teacher(s_k,a)
pi_teacher(a|I) = segment_softmax(Q_bar / temperature)
loss = KL(pi_teacher || pi_student) + c_value * Huber(mc_q_student, Q_bar)
```

Sampling defaults to `K=4`, temperature `0.5`, and stopped gradients through the
discrete Belief samples. The true-state-only direct KD function remains as an
explicit ablation. Tests require all sampled states to conserve cards, each
Teacher target segment to sum to one, Student-only finite gradients, and a
different averaged target from direct-state KD.

Configuration is frozen in
[`../configs/model/privileged_teacher_v1.yaml`](../configs/model/privileged_teacher_v1.yaml)
and [`../configs/train/is_kd.yaml`](../configs/train/is_kd.yaml). The default
Teacher has 24,124,492 parameters. M6 unit/smoke gates establish correctness and
leakage isolation; claims that Teacher or IS-KD improves playing strength require
the fixed-deal experiment and direct-vs-IS ablation, not these mechanical tests.
