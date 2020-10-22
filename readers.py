import re

from collections import OrderedDict

SPLIT_KEEP_WHITESPACE_RE = re.compile(r'(\s+)')


def parse_fasttext_line(l, label_string='__label__'):
    split, idx = SPLIT_KEEP_WHITESPACE_RE.split(l), 0

    # Skip initial whitespace or empties
    while idx < len(split) and split[idx].isspace() or not split[idx]:
        idx += 1
    
    # Collect and skip labels
    labels = []
    while idx < len(split) and split[idx].startswith(label_string):
        labels.append(split[idx][len(label_string):])
        idx += 1
        # Skip whitespace and empties
        while idx < len(split) and split[idx].isspace() or not split[idx]:
            idx += 1

    # The rest is the text
    text = ''.join(split[idx:])
    return labels, text


def read_fasttext(f, fn, label_string='__label__'):
    for ln, l in enumerate(f, start=1):
        l = l.rstrip('\n')
        try:
            labels, text = parse_fasttext_line(l, label_string)
        except Exception as e:
            raise ValueError(f'failed to parse {fn} line {ln}: {e}: {l}')
        yield text, labels


READERS = OrderedDict([
    ('fasttext', read_fasttext)
])


def get_reader(format_name):
    try:
        return READERS[format_name]
    except KeyError:
        raise ValueError(f'unknown format {format_name}')
