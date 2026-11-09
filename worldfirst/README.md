# World-first, stage 1: a model that learns the laws of a small world

**Logos1 (the Laws model) is not a next-word model.** It never learns to continue text. It reads a scene, sees an action, and
predicts what is true afterwards, and it learns only from being wrong about that. Its rival **is** a plain next-word
model of the same size, trained on the very same world text.

The question stage 1 answers: on cases neither model has ever been shown (longer sums, deeper boxes, family links
chained in new ways, machines mixing rules never mixed), does the one built to find laws get them right, where the
one built to continue text does not? That is the make-or-break for the whole plan.

## Run it

```bash
bash runpod/world.sh                                   # on the pod: trains both, examines both, writes the report
python3 -m worldfirst.run                              # the same, in this terminal
python3 -m worldfirst.run --preset tiny --steps 300 --eval-n 20   # a quick CPU check (minutes)
python3 -m worldfirst.tests                            # the world's walls and both models (about a minute, CPU)
bash runpod/world.sh --name world-big --preset big --steps 200000 --passes 3   # the long run: ~29M each, overnight
```

On a GPU both models train at the same time (one process each), and helper processes build the batches. Each model
saves its state every 10 minutes: if the pod stops, the same command carries on where it was.

No dataset, no tokenizer download: the world writes its own text. Results land in `<runs>/worldfirst/`:
`report.md` (the tables in words), `results.json` (every number), `laws/` and `rival/` (checkpoints and training
logs). The dashboard shows them as the "Understanding" card. `--only report` examines the saved models again.

`--passes 3` makes the core always think exactly 3 passes, from a blank start, learning through all of them. The
architecture test showed why: without loops, longer sums fell from 76% to 29% (carrying needs several ticks), but
the random 1 to 16 passes made lines and boxes harder to learn. A few fixed passes aims for both.

Options worth knowing: `--steps` (20,000 each), `--batch` (128 episodes), `--preset tiny|small|base|big`,
`--no-curriculum`, `--rival-tags 0` (the rival without the "same thing" marks), `--rival-loss answers` (a rival
that learns only the answers: a stronger, non-plain variant to check against), `--minutes` (a time cap per model).

The no-marks race (after the big run): the world speaks more naturally (she/he/her/him; sums smallest digit first,
for both), the rival gets no "same thing" tags, and the Laws model finds the things itself with 32 slots (slot
attention); a tiny speaker writes the sums. About 38M each (the finder and speaker add to the Laws model; the rival is
widened to match). The report adds how many things it found cleanly.

```bash
bash runpod/world.sh --name world-natural --preset big --steps 200000 --passes 3 --natural 1
```

Marks as a teacher (`--teach 1`): the same no-marks race, but in training only the world's marks are the answer key
for the finder ("these words are the same thing"). It is graded on which words it groups together, never given the
marks as input, so at the exams (and in use) it reads plain text like the rival. The rival is unchanged.

```bash
bash runpod/world.sh --name world-teach --preset big --steps 200000 --passes 3 --natural 1 --teach 1
```

The linker (`--linker 1`, the best no-marks version): instead of slots, each word that names a thing points back to
the earlier word that names the same thing, or starts a new thing; the chains are the things, numbered in the order
it met them. The marks are only its answer key in training (as `--teach`); it always uses its own links.
For fairness the rival gets the very same answer key the same way: a link head on its last layer, the same loss and
weight, sizes still matched. At the exams both read plain text.

```bash
bash runpod/world.sh --name world-link --preset big --steps 200000 --passes 3 --natural 1 --linker 1
```

A looped rival (`--rival-loops 3`): the same rival, but its last 2 layers run 3 times with the same weights, like
Logos1's core. Same size, about 1.6x the work per step. `--laws-from world-link` trains only this rival and reuses
world-link's Logos1 for the report (if world-link is still training, rerun with `--only report` when it is done).

```bash
bash runpod/world.sh --name world-loop --preset big --steps 200000 --passes 3 --natural 1 --linker 1 --rival-loops 3 --laws-from world-link --rival-ckpt 1
```

## The world (`world.py`)

Each episode is a scene in simple English with fresh random names and colours, one action, and two or three
questions about the scene *after* the action. The world computes the true new scene, so every answer is right.
Each kind of link is said several ways ("Ana stands just in front of Bo", "Bo is right behind Ana").

| family | the law | an action | seen in training | secret exams |
|---|---|---|---|---|
| sums | adding with carries, digit by digit | "add them" | 1 to 8 digits | 9 to 12 digits |
| lines | order: in front of / behind, chained | swap two, one goes to the front or back; 1 or 2 moves | 3 to 8 people | 9 to 12 people; 3 or 4 moves in a row |
| boxes | inside of, nested | take a box out, put one into another; 1 or 2 moves | 1 to 6 boxes deep | 7 to 9 deep; 3 or 4 moves in a row |
| family | parents and children, brothers, sisters, said one way and asked either way | a baby is born | one, two or three links chained (31 of 36 pairs) | the 5 pairs never asked, in two or three links; four links |
| falling | a block falls until it rests on something | a nailed block comes loose | 1 to 4 blocks fall | 5 to 7 blocks fall |
| machines | a hidden rule, learned from a few tries | try new numbers | one step, or two steps (most pairs) | 6 pairs of steps never mixed; three steps |
| cups (hidden) | a thing goes on existing when hidden, and moves with its cup | cups swap by place or by colour; a thing moved to another cup | 1 to 4 moves | 6 to 8 moves |
| minds | people believe what they last saw, not what is true now | people leave and come back; things are moved while they are away | 1 to 3 moves | 4 to 6 moves |
| walks | where you are and which way you face | turn left, right or around, walk some steps | 1 to 4 legs | 5 to 7 legs |

The walls are built in, not hoped for: training only ever asks for training levels, the held-out family pairs and
machine mixes are never generated for training (a held-out machine rule is one that no training rule equals), and
`tests.py` checks thousands of training episodes against every secret condition. Fresh random names make
memorising scenes useless; only the laws carry over.

A curriculum serves each learner the edge of what it can do: per family, it starts at level 1 and moves up when the
learner gets 85% right there; half its episodes come from the newest level and half from the ones before. Each model
gets its own curriculum, by the same rule.

## Logos1, the Laws model (`model.py`)

1. **Reader**: a small transformer (4 layers) reads the scene, the action and the questions, never the answers.
2. **One vector per thing**: the world marks which words name the same thing (like pointing at it), and those words
   are pooled into that thing's vector. Each question gets a vector too.
3. **Laws core**: a tiny shared network (2 layers) where the things look at each other and update. It runs again and
   again: each **pass** is one tick of thinking, and the picture sharpens. It is fed the reader's picture again on
   every pass, so it never loses the scene. It only ever sees things, never words. Things keep their order of
   appearance (rotary positions on where each was first named), so "the column to my right" means the same in a
   9-digit sum as in a 3-digit one.
4. **Head**: answers from the settled picture: yes/no or a digit from a question's vector, "who/what" by pointing
   from the question to a thing, a sum by reading one digit from each column.

**Training**: a random number of passes each step (1 to 16), learning only through the last 4, each starting from
random noise (Huginn's recipe, at toy scale). So it must work at any depth, and long thinking costs no extra memory.
**Testing**: it thinks until its picture stops moving (a pass changes it by less than 1%), up to 48 passes. That
number is its own choice, and the report shows it against difficulty: if it truly reasons, harder scenes take more.

About 15M numbers at the base size (d 448, 7 heads), about 29M at the big size (d 576, 9 heads, a 5-layer reader).

## The rival

A normal causal transformer, 6 layers at the same width, its MLP sized so it has the same number of numbers (within
1%). It learns by guessing every next word of the same episodes, answers included (`--rival-loss answers` is the
variant that learns only the answers). It gets the same help with names: each thing's words carry a tag that is
random every episode, so the tag only says "same thing". It is asked the same questions: its guess right after
"answer :" among the same candidates (yes/no, digits, or the things in the scene). A sum counts only if every digit
is its best guess, which is the same as writing the whole sum right.

What it does not get is loops: 6 layers of thinking per word, always. That is the difference under test.

## How it is judged

- **Secret exams vs practised kinds**, per family and level, both models, a question right only if its whole answer is.
- **Passes against difficulty** (the Laws model's own choice when to stop), and accuracy at 1, 2, 4, ... 32 forced
  passes: does thinking longer help, and does it hold up past the 16 it trained with?
- **Reversals**: family questions asked the other way round from how the link was told ("Rui is the son of Paulo",
  then "who is the father of Rui?"), against questions asked the way it was told.

What stage 1 does not do yet: no notebook, no planner, no talking. The plan's next stages build on this one only if
the Laws model wins the secret exams.
