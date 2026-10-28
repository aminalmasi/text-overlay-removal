# The Gemini half: what each prompt asks and returns

Every photo gets one cheap gate call; at most one more call follows. All three prompts run on
`gemini-3.5-flash-lite` with the same settings: the prompt as the system instruction, the photo as the only content,
a strict JSON response schema, temperature 0, thinking LOW, default image resolution, flex tier.
[`tl_gemini.py`](tl_gemini.py) makes the calls, retries and falls back exactly as the tested runs did.

```
photo ─► gate ─┬─ removal 1 ─────────────► worker ─► text + boxes ─► GPU eraser
               ├─ eligible, removal 0 ───► text prompt ─► text only (nothing to erase)
               └─ neither ───────────────► done (no text)
any failed call ─► the marketplace's existing behaviour: text prompt alone, nothing erased
```

The prompt texts and schemas are not published. This page describes what each one asks for and every field it
returns, which is all the eraser depends on.

## Usage per call

| Call | Runs on | Input tokens (photo + prompt) | Output tokens | Latency, median (p90) |
|---|---|---:|---:|---:|
| Gate | every photo | 1,080 + 408 | 12 | 2.1 s (3.2) |
| Worker | photos with `removal: 1` | 1,090 + 1,605 | 455 | 1.7 s (3.9) |
| Text prompt | photos with `eligible: true, removal: 0` | 1,085 + 228 | 130 | not logged |

The photo is about 1,080 tokens whatever the prompt, so the gate's cost is mostly the photo itself. Output tokens
cost about 8× more than input, which is why the gate answers in two fields and nothing else.

## Gate: two questions about every photo

The gate answers two independent questions.

**`eligible`** (boolean): does the picture carry *digital* text? This means seller-added overlays (prices,
currencies, phone numbers, links), infographics, screenshots of specification pages and screenshots of messages to
buyers. Writing on physical things never counts: printing, engraving, logos, badges, stickers, licence plates,
documents on or beside the product. A brand printed on a box is excluded; the same brand typed on as an overlay is
included. This keeps the meaning of the field the marketplace already stores, so the gate can take over that call.

**`removal`** (0 or 1): was text, an emoji, a sticker or a watermark added *on top of a real photograph*, so
that removing it leaves a hole to reconstruct? The prompt lists what is 0 even though it carries text:
- text on its own flat band, header or panel beside the photographs (a typed description joined to a photo grid);
- infographics, posters and designed adverts built around their words;
- products on a studio or cut-out background;
- screenshots, and what apps draw over photos (`+3` counters, likes, play buttons, status bars);
- writing that belongs to the photographed world.

A caption on the plain padding bar of a single photo is 1. Two rules settle the hard cases: "the amount of
writing is not evidence" and "when unsure, 0".

```json
{"eligible": true, "removal": 1}
```

How the prompt is built:
- it states the contrast first (flat digital text against writing on objects), then lists what counts;
- it gives explicit lists of what does not count;
- it ends with a one-line summary of the four outcomes;
- it tells the model to treat image content as data, never as instructions.

## Worker: the text, and a box for every added mark

The worker runs only when the gate says `removal: 1`. It does two jobs in one reply.

**Text.** It copies only the added text, verbatim, in reading order, one piece per line. It never translates,
summarises, describes objects or guesses unreadable text, and it uses the same "digital, not physical" rule as the
gate.

**Regions.** A region is added content (text, emoji, sticker, badge, seal, watermark) lying on a photograph, or
burned onto the plain padding bar of a single photo. Text on its own band, in an infographic or in a screenshot is
copied but not boxed, and app-drawn interface elements are never regions. The rules that took the longest to get
right:
- **One region is one mark.** A logo and the words that go with it are one region. Emoji and symbols at either end
  of a line belong to that line's region. Lines sharing one carrier are one region. Lines with photograph showing
  between them are separate regions: three lines over a gate are three boxes, never one. Every copy of a
  watermark is reported, however faint.
- **The box covers the carrier.** `box_2d` runs to the outer edge of whatever the text sits on (panel, band,
  chip, speech bubble, sticker body, gradient scrim, outline, glow, shadow), not just the letters. The prompt
  lists the carriers models miss most: a panel close in colour to the photo, a light panel on a pale background,
  a scrim that fades out, a bar running off the frame.
- **Protect what must stay.** Whenever there is something to erase, the worker also reports each protected
  surface as its own region with `physically_in_scene: true`: licence plates, chassis and engine plates, printed
  serials and model numbers, identity documents, screens showing content, meters, and prices or weights on a
  device's own display. A mark pasted over a plate is fitted to the mark alone, and the plate is reported too.
- **Self-check.** Before answering, every copied line that lies on a photograph must sit inside some region's box;
  if one does not, the model adds that region.

### Output fields

Coordinates are integers `[ymin, xmin, ymax, xmax]` on a 0–1000 grid over the full image.

| Field | Values | Used by the eraser for |
|---|---|---|
| `text` | all added text, one piece per line | stored with the listing (not used for erasing) |
| `boxes[]` | one object per region | |
| `box_2d` | the region including its carrier | the SAM2 prompt and the mask region |
| `text_box_2d` | the readable content alone | finding leftover text in layer 2 |
| `line_boxes` | one box per line, inside `text_box_2d` | splitting bare text into lines |
| `carrier_present` | true / false | bare text is split and refined line by line; a carrier is erased whole |
| `carrier_kind` | none, panel, bubble, chip, band_ribbon, sticker_body, gradient_scrim, outline_shadow | diagnostics |
| `carrier_opacity` | none, translucent, opaque | diagnostics |
| `kind` | text, emoji_symbol, logo_watermark, sticker_badge, mixed | box refinement (text only; emoji regions are widened) |
| `text` (per region) | that region's text, verbatim | diagnostics |
| `confidence` | certain, probable, possible | diagnostics |
| `physically_in_scene` | false = composited on afterwards; true = part of the scene, or unsure | erase (false) or protect (true) |

A region whose `physically_in_scene` is missing or malformed is protected, never erased.

What the 455 output tokens are: field names repeated in every region (about 113 tokens) and coordinates (about 92;
every digit is its own token, so one box costs about 20). Three shorter reply formats were tested; all three boxed
worse, so the reply stays as it is.

## Text prompt: the marketplace's existing call

Photos with text but nothing to erase (screenshots, infographics, typed bands) go to the marketplace's existing
text-extraction prompt, unchanged. It returns the same two fields the marketplace already stores, `eligible` and
`text`. It is also the fallback: if the gate or the worker fails after its retries, this call alone decides, as it
did before. If it fails too, the fields come back unknown (`None` with an `error`), never as a made-up "no text".

Sending those photos to the worker instead would box every line of a screenshot (about 1,400 output tokens) only
for the boxes to be thrown away; the text prompt copies the same text for about a sixth of the cost.

## How the prompts got here

| Version | What changed | Outcome |
|---|---|---|
| P61 | one call boxed the overlays | the first end-to-end pipeline (August) |
| P80–P83 | text and removal combined in one or two calls | a single combined call (P82) dropped screenshot text |
| P84 | the lean gate: two fields, `removal` as a bare 0/1 | kept; the per-photo prompt cut by more than half |
| P86, P87 | shorter worker replies | boxed worse; rejected |
| P88, P91 | earlier workers | boxes too broad on some photos |
| P92 | line boxes, the carrier rules, protected surfaces | kept: the final worker |
| P93, P94 | lines first, then regions | missed more regions; P94 erased a logo together with a plate |
