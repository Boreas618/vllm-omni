"""Assemble independently authored scene briefs at exact tokenizer lengths.

The scene and photography prose is authored in full, not generated from object
pair templates. Selection stops at complete sentences. A concrete prop detail
distinguishes each request; seeds, measurements and generated outputs do not
influence wording or selection.
"""

import hashlib
import itertools
import json
import re
from functools import lru_cache
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'workload'
MODEL = Path('/workspace/models/FastVideo-FastH3-8-Step-V2')
TOKENIZER = Tokenizer.from_file(str(MODEL / 'tokenizer/tokenizer.json'))
REVISION = (MODEL / '.cache/huggingface/download/text_encoder/tokenizer.json.metadata').read_text().splitlines()[0]
FAMILIES = ('harbor', 'ceramics', 'percussion', 'river', 'observatory', 'market')
COLORS = ('ochre', 'slate blue', 'forest green', 'burgundy', 'warm gray', 'indigo', 'cream', 'rust red')
DETAILS = (
    'a plain surface', 'a narrow border', 'a fine diagonal weave',
    'a broad central stripe', 'two thin parallel stripes', 'a small stitched corner',
    'a subtle herringbone texture', 'a darker reinforced edge',
    'a small square patch near one end', 'a lightly faded center',
    'a simple crosshatched texture', 'a row of small evenly spaced stitches',
)


@lru_cache(maxsize=8192)
def encode(value):
    return TOKENIZER.encode(value, add_special_tokens=False).ids


def variation(family, variant):
    color = COLORS[variant % len(COLORS)]
    detail = DETAILS[variant // len(COLORS)]
    subject = {
        'harbor': 'The fabric cover around the folded tide chart',
        'ceramics': "The apprentice's apron pocket",
        'percussion': "The drummer's woven wristband",
        'observatory': 'The cloth cover of the upper field notebook',
        'market': "The customer's cloth shoulder bag",
    }
    if family == 'river':
        shapes = ('rounded', 'flat-topped', 'slightly oval', 'low and broad', 'narrow and sloping', 'gently domed', 'asymmetrical', 'shallowly ridged')
        leaves = ('a curled beech leaf', 'a narrow willow leaf', 'a small oak leaf', 'a pale birch leaf', 'a dark alder leaf', 'a broad maple leaf', 'a dry hazel leaf', 'a mottled elm leaf', 'a torn chestnut leaf', 'a folded hornbeam leaf', 'a small lime leaf', 'a weathered ash leaflet')
        return f'The foreground mossy stone is {shapes[variant % 8]}. Beside its dry upper edge lies {leaves[variant // 8]}, lodged securely in a shallow crease above the waterline. This small stationary detail remains peripheral to the bird and its dive.'
    return f'{subject[family]} is {color}, with {detail}. This detail belongs to the existing object, stays at its established location and remains subordinate to the main action. Preserve its material and appearance through the camera movement.'


def sentences(value):
    # Paragraphs containing dialogue remain intact so a quotation is never
    # separated from its speaker or the scene's core action.
    for paragraph in value.split('\n\n'):
        paragraph = paragraph.strip()
        if not paragraph or paragraph.startswith('#'):
            continue
        if '"' in paragraph:
            yield paragraph + '\n'
        else:
            yield from (sentence + '\n' for sentence in re.split(r'(?<=[.!?])\s+', paragraph))


def prompt(family, variant, count):
    scene = (ROOT / 'scenes' / f'{family}.md').read_text()
    first, rest = scene.split('\n\n', 1)
    introduction = first + '\n' + variation(family, variant) + '\n'
    parts = [introduction, *sentences(rest), *sentences((ROOT / 'scenes/photography.md').read_text())]
    # Complete, meaningful closing directions cover the small remainder at
    # the token boundary. No text is duplicated to increase prompt length.
    endings = [f'{verb} {subject}{tail}.' for verb, subject, tail in itertools.product(
        ('Preserve', 'Keep', 'Maintain'),
        ('the established framing', 'the natural light', 'the scene continuity', 'the quiet final composition', 'the environmental sound'),
        ('', ' through the ending', ' while the main action settles', ' and let the ambient sound continue', ' without adding another event', ' as the scene reaches its stated resolution'),
    )]
    prefixes = ('', 'Let the scene end naturally. ', 'Keep the final gesture restrained and preserve the surrounding atmosphere. ', 'Allow the last small movement to settle within the established composition, with the environmental sound continuing around it. ')
    closing = [prefix + ending for prefix, ending in itertools.product(prefixes, endings)]
    by_length = {}
    for value in closing:
        by_length.setdefault(len(encode(value)), value)
    text = ''
    length = 0
    for part in parts:
        candidate = text + part
        approximate = length + len(encode(part))
        if approximate > count:
            break
        if count - approximate < 80:
            for closing_length, ending in by_length.items():
                if abs(closing_length - (count - approximate)) <= 2:
                    finished = candidate + ending
                    if len(encode(finished)) == count:
                        return finished
        text = candidate
        length = approximate
        if length == count and len(encode(text)) == count:
            return text
    raise ValueError(f'Insufficient complete prose at {family}/{variant}/{count}: {length}')


def main():
    OUTPUT.mkdir(exist_ok=True)
    if (OUTPUT / 'authored-corpus.jsonl').exists():
        raise FileExistsError('The authored corpus already exists')
    rows = []
    variants = [0] * 6

    def add(partition, family, seconds, tokens, seed=None, **extra):
        variant = variants[family]
        variants[family] += 1
        name = FAMILIES[family]
        value = prompt(name, variant, tokens)
        frames = round(seconds * 24)
        rows.append({
            'id': f'{partition}-{name}-{variant:03}', 'prompt': value,
            'prompt_len': tokens, 'seconds': seconds,
            'seed': seed if seed is not None else 1101 + variant,
            'metadata': {
                'partition': partition, 'family': name, 'variant': variant + 1,
                'prompt_sha256': hashlib.sha256(value.encode()).hexdigest(),
                'tokenizer_repository': 'FastVideo/FastVideo-FastH3-8-Step-V2',
                'tokenizer_revision': REVISION, 'tokenizer_subfolder': 'tokenizer',
                'token_ids': encode(value), 'aligned_frames': frames + (5 - frames) % 17,
                'provenance': {'kind': 'independently_authored', 'source_url': None, 'case': name,
                    'expansion': 'Authored continuous-scene production brief: spatially consistent setting, contact mechanics, material and optical detail, performance, environmental sound and camera continuity.'},
                **extra,
            },
        })

    shapes = list(itertools.product((5, 10, 15), (1000, 10000)))
    for family in range(6):
        for seconds, tokens in shapes:
            for seed in (1101, 2202):
                add('latency', family, seconds, tokens, seed=seed)
    for seconds, tokens in shapes:
        for index in range(16):
            add('throughput', index % 6, seconds, tokens)
    for partition in ('warmup', 'qualification'):
        for family, (seconds, tokens) in enumerate(shapes):
            add(partition, family, seconds, tokens)
    for concurrency in (1, 2, 4, 8, 16):
        for repetition in (1, 2, 3):
            for index in range(2 * concurrency):
                seconds, tokens = shapes[index % 6]
                add('priming', (index + repetition) % 6, seconds, tokens, concurrency=concurrency, repetition=repetition)
    (OUTPUT / 'authored-corpus.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    provenance = {
        'tokenizer_repository': 'FastVideo/FastVideo-FastH3-8-Step-V2', 'revision': REVISION,
        'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (MODEL / 'tokenizer').iterdir() if p.is_file()},
        'token_count_policy': 'raw prompt, no added special tokens',
        'authoring_sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), *sorted((ROOT / 'scenes').glob('*.md'))]},
    }
    (OUTPUT / 'tokenizer.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Authored {len(rows)} unique requests; family counts={variants}')


if __name__ == '__main__':
    main()
