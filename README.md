# Graphglyph

Encode text as reversible blue-and-orange unit-distance graph images.

The visual style is based on the finite illustration from OpenAI's unit-distance
paper: points of the form

```text
z = a + b i + c rho + d i rho
rho = exp(2 pi i / 3)
```

with edges drawn between projected points at Euclidean distance `1`. See
[Planar Point Sets with Many Unit Distances](https://cdn.openai.com/pdf/74c24085-19b0-4534-9c90-465b8e29ad73/unit-distance-proof.pdf).

## How It Works

- Text is normalized with Unicode NFKC and encoded as UTF-8.
- Long payloads are zlib-compressed.
- A packet header stores magic bytes, version, flags, seed, length, and CRC32.
- Payload nibbles are distributed through seeded graph cells.
- Each cell encodes four bits by making one edge in each candidate edge pair
  stronger than the other.
- The same text seed also varies the coefficient window, point count, basis
  family, edge weights, and payload placement.

The SVG and JSON outputs are decodable. PNG outputs are presentation previews.

## Usage

```bash
python3 graph_cipher.py encode "you are loved immensely" \
  -o examples/you_are_loved_immensely.svg \
  --json examples/you_are_loved_immensely.json

python3 graph_cipher.py decode examples/you_are_loved_immensely.svg
```

Colors are presentation-only and do not affect decoding:

```bash
python3 graph_cipher.py encode "goblins" -o goblins_dark.svg \
  --edge-color "#7c8cff" \
  --node-color "#ffd166" \
  --node-stroke-color "#d08700" \
  --background-color "#0b1020"
```

Use `--variant-strength 0` for the exact finite algebraic box. The default is
`0.75`, which gives more visible variation between texts.

## Examples

### You Are Loved Immensely

[SVG](examples/you_are_loved_immensely.svg) |
[PNG](examples/you_are_loved_immensely.png)

![you are loved immensely](examples/you_are_loved_immensely.png)

### Goblins

[SVG](examples/goblins.svg) |
[PNG](examples/goblins.png)

![goblins](examples/goblins.png)

### Meditations On Moloch

Encoded from Scott Alexander's
[Meditations On Moloch](https://slatestarcodex.com/2014/07/30/meditations-on-moloch/).

[PNG](examples/meditations_on_moloch.png)

![Meditations On Moloch](examples/meditations_on_moloch.png)

The large `meditations_on_moloch.svg` is intentionally not included.

All code written by GPT 5.5.
