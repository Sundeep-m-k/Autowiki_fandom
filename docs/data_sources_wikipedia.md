# Data Sources — Wikipedia

Technical reference for acquiring, parsing, and integrating Wikipedia data into the
Autowiki pipeline. Use this document alongside the README for implementation details
and for paper writing.

---

## 1. Why Wikipedia?

The Fandom pipeline acquires data by crawling individual wiki pages. This works well
for domain-specific wikis (hundreds to low thousands of articles), but does not scale
to Wikipedia's ~7 million English articles due to:

- **Rate limits**: Wikipedia enforces strict request throttling on its API and live pages.
  Crawling 7M pages at 2 s/request would take ~162 days.
- **IP bans**: Automated crawling without the Wikimedia Enterprise tier risks
  temporary or permanent IP blocks.
- **Reproducibility**: Live pages change; a crawl is a snapshot with no stable version.

**Solution:** Wikimedia publishes full, versioned XML dumps of every wiki at
`https://dumps.wikimedia.org/`. These are compressed multi-gigabyte files that can be
downloaded once and re-processed offline as many times as needed.

---

## 2. Dump Format

### 2.1 File type

Wikipedia dumps are distributed as `.xml.bz2` files. The main article dump for English
Wikipedia is split into ~100 chunks named:

```
enwiki-latest-pages-articles1.xml-p1p41242.bz2     # pages 1–41,242   (~280 MB)
enwiki-latest-pages-articles2.xml-p41243p151573.bz2 # pages 41,243–151,573
...
```

Chunk 1 (`p1p41242`) is the one currently used by this project. It is available at:

```
https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles1.xml-p1p41242.bz2
```

The file is stored at `data/raw/wikipedia/` in this repository (not committed to git
because of size; re-download as needed).

### 2.2 XML structure

Each article is a `<page>` element inside a `<mediawiki>` root:

```xml
<page>
  <title>Anarchism</title>
  <ns>0</ns>                   <!-- namespace: 0 = main article space -->
  <id>12</id>
  <revision>
    <text>
      '''Anarchism''' is a [[political philosophy]] and [[Political movement|movement]]...
    </text>
  </revision>
</page>
```

The `<text>` element contains **wikitext** — the raw markup language used by all
MediaWiki-based wikis (Wikipedia, Fandom, etc.). Internal links are expressed as
`[[Target Page|Anchor text]]` or `[[Target Page]]`.

### 2.3 What chunk 1 contains

| Count | Category |
|-------|----------|
| ~41,242 | Total `<page>` entries (including redirects) |
| ~21,009 | True articles (namespace 0, not a redirect) |
| ~20,233 | Redirect pages (skipped by the parser) |

---

## 3. Why XML Dumps and Not Enterprise HTML?

Three alternatives were considered:

| Option | Description | Decision |
|--------|-------------|----------|
| **Live web crawling** | `requests` + `BeautifulSoup` against `en.wikipedia.org` | Rejected — rate limits, scale |
| **Enterprise HTML dumps** | Pre-rendered HTML from `dumps.wikimedia.org/other/enterprise_html/` | Rejected — archive is historical (not updated), live access requires paid Wikimedia Enterprise account |
| **Wikipedia REST API HTML** | `GET /api/rest_v1/page/html/<title>` per article | Rejected — severe bulk rate limits; HTML structure uses `<section>` tags, not `mw-parser-output`, breaking existing parser |
| **XML dumps + `mwparserfromhell`** | Parse wikitext offline | **Chosen** |

**Advantages of XML dumps:**

- **Offline and reproducible**: Download once, process any number of times.
- **Complete**: Every article in the dump, including stubs and short articles.
- **Clean ground truth**: Wikitext links (`[[Target|Anchor]]`) are explicit and unambiguous.
  There is no risk of extracting nav-bar links, footer links, or infobox links by mistake —
  a problem that HTML-based parsers must work around with heuristics.
- **Versioned**: Each dump has a date stamp, enabling reproducible experiments.
- **No rate limits**: The bz2 file is downloaded in a single HTTP request.

---

## 4. Parser Design

### 4.1 Two-pass streaming

The dump is streamed with `xml.etree.ElementTree.iterparse` — the bz2 file is never
fully decompressed into memory. `elem.clear()` is called after each `<page>` is
processed to keep memory usage constant regardless of file size.

**Pass 1** indexes all articles in the dump to build the mapping:

```
page_name (title with spaces→underscores) → page_id (int)
```

This mapping is needed in Pass 2 so that when we encounter `[[Political movement|movement]]`
inside an article, we can immediately record `article_id_of_internal_link` for "Political
movement" without making any further disk I/O.

**Pass 2** processes each article's wikitext and emits JSONL records.

### 4.2 Wikitext parsing with `mwparserfromhell`

`mwparserfromhell` is a C-accelerated Python library that parses wikitext into a typed
node tree. The node types used by this pipeline are:

| Node type | What it is | Action |
|-----------|-----------|--------|
| `Text` | Plain text segment | Append to output text, advance offset |
| `Wikilink` | `[[Target\|Anchor]]` | Append anchor text, record link + offsets |
| `Template` | `{{Infobox ...}}`, `{{Cite ...}}` | **Drop** (noise removal) |
| `Tag` | `<ref>`, `<br>`, etc. | Drop `<ref>` content; include other tag text |
| `Heading` | `== Section ==` | Skip (heading text is part of structure, not prose) |

**Noise removal**: Before the node walk, all templates whose names start with known
noise patterns are removed in-place from the parsed wikicode. The full list is in
`src/data_processing/wikipedia_ground_truth.py → NOISE_TEMPLATES`.

### 4.3 Character offset tracking

Plain text and link positions are built in a single pass over the node list:

```
offset = 0
for node in wikicode.nodes:
    if Text node:
        output_text += node.text
        offset += len(node.text)
    if Wikilink node:
        anchor = node.text or node.title
        link_start = offset
        output_text += anchor
        offset += len(anchor)
        record link(start=link_start, end=offset, ...)
```

This means offsets are exact character indices into the plain text — no post-hoc regex
alignment needed. The same approach is used in `ground_truth.py` for Fandom HTML
(via BeautifulSoup's `.strings` iteration), so the output schema is structurally identical.

### 4.4 Paragraph and sentence splitting

After the node walk, the full plain text is split into paragraphs on double newlines
(`\n\n`). Each paragraph's links are re-projected from global offsets to paragraph-local
offsets.

Sentences are split from each paragraph using NLTK's Punkt tokenizer
(`nltk.sent_tokenize`), which handles abbreviations, ellipses, and similar edge cases.
Sentence links are re-projected from paragraph-local to sentence-local offsets using the
same `_links_in_segment()` function as the Fandom pipeline.

### 4.5 Link classification

All wikitext internal links (`[[...]]`) are classified as `link_type: "internal"` unless
the target starts with a non-article namespace prefix (see `SKIP_PREFIXES` in
`wikipedia_ground_truth.py`):

```python
SKIP_PREFIXES = (
    "File:", "Image:", "Template:", "Category:", "Wikipedia:",
    "Talk:", "User:", "Help:", "Special:", "Portal:", ...
)
```

Skipped-prefix links still contribute their display text to the plain text (so character
offsets remain valid) but are not recorded as links. This mirrors how Fandom's
`_classify_link()` treats category and file links.

---

## 5. Output Schema

The output schema is **identical** to the Fandom ground truth pipeline. No changes are
needed in Tasks 1, 2, or 3 to use Wikipedia data — just point the config at
`domain: "wikipedia"`.

### Paragraph record

```json
{
  "granularity": "paragraph",
  "article_id": 12,
  "page_name": "Anarchism",
  "title": "Anarchism",
  "paragraph_id": "wikipedia_paragraph_0000001",
  "paragraph_index": 0,
  "paragraph_text": "Anarchism is a political philosophy and movement...",
  "url": "https://en.wikipedia.org/wiki/Anarchism",
  "source_path": "data/raw/wikipedia/12.xml",
  "links": [
    {
      "anchor_index": 1,
      "anchor_text": "political philosophy",
      "plain_text_rel_char_start": 15,
      "plain_text_rel_char_end": 35,
      "link_type": "internal",
      "target_page_name": "political_philosophy",
      "article_id_of_internal_link": 23456,
      "resolved_url": "https://en.wikipedia.org/wiki/political_philosophy"
    }
  ]
}
```

### Article-page-granularity record

```json
{
  "granularity": "article",
  "article_id": 12,
  "article_record_id": "wikipedia_article_0000012",
  "page_name": "Anarchism",
  "title": "Anarchism",
  "article_plain_text": "Anarchism is a political philosophy...\n\n...",
  "url": "https://en.wikipedia.org/wiki/Anarchism",
  "source_path": "data/raw/wikipedia/12.xml",
  "links": [
    {
      "anchor_index": 1,
      "anchor_text": "political philosophy",
      "link_type": "internal",
      "plain_text_char_start": 15,
      "plain_text_char_end": 35,
      "target_page_name": "political_philosophy",
      "article_id_of_internal_link": 23456,
      "resolved_url": "https://en.wikipedia.org/wiki/political_philosophy"
    }
  ]
}
```

---

## 6. Differences from Fandom Pipeline

| Aspect | Fandom | Wikipedia |
|--------|--------|-----------|
| Acquisition | Live HTTP scrape (`requests`) | Offline XML dump (bz2) |
| Input format | Rendered HTML | Raw wikitext |
| Parser | `BeautifulSoup` (HTML) | `mwparserfromhell` (wikitext) |
| Article ID | `page_id` from MediaWiki API | `<id>` element from XML |
| Metadata | Extracted from JavaScript vars in HTML (`wgPageName`, `wgTitle`) | Extracted from XML `<title>` and `<id>` elements |
| Base URL | `https://<domain>.fandom.com/wiki/` | `https://en.wikipedia.org/wiki/` |
| Link resolution | HTML `<a href>` attributes | `[[Target\|Anchor]]` syntax |
| Script | `00_scrape_fandom.py` + `01_build_ground_truth.py` | `00_parse_wikipedia_dump.py` |
| Config | `ground_truth.yaml` | `wikipedia_ground_truth.yaml` |
| Output schema | Identical | Identical |

The key design principle is that **the output is source-agnostic**. Tasks 1–3 see only
JSONL records with `article_id`, `paragraph_text`, `links`, etc. They have no knowledge
of whether the data came from Fandom or Wikipedia.

---

## 7. Extending to More Chunks / Languages

To add more Wikipedia data, download additional chunk files and run the parser with
`--domain` to keep outputs separate:

```bash
# Download chunk 2
wget -c "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles2.xml-p41243p151573.bz2" \
     -O data/raw/wikipedia/enwiki-latest-pages-articles2.xml-p41243p151573.bz2

# Parse chunk 2
python3 scripts/01_Data_processing/00_parse_wikipedia_dump.py \
    --dump data/raw/wikipedia/enwiki-latest-pages-articles2.xml-p41243p151573.bz2 \
    --domain wikipedia_chunk2
```

For non-English Wikipedias, replace `enwiki` with the language code (e.g. `dewiki`,
`frwiki`) in the dump URL. The parser is language-agnostic.

---

## 8. Files Added

| File | Purpose |
|------|---------|
| `src/data_processing/wikipedia_ground_truth.py` | Core parsing module |
| `scripts/01_Data_processing/00_parse_wikipedia_dump.py` | Runnable CLI script |
| `configs/data_processing/wikipedia_ground_truth.yaml` | Configuration |
| `data/raw/wikipedia/enwiki-latest-pages-articles1.xml-p1p41242.bz2` | Downloaded dump (not in git) |
| `data/processed/wikipedia/` | Output JSONL and CSV files (not in git) |
| `data/stats/wikipedia.json` | Dataset statistics |
