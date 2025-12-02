# DBLP Data Dictionary

This document provides comprehensive documentation for all tables, columns, and derived metrics in the DBLP dataset after ETL processing.

---

## Overview

The DBLP dataset is processed into four primary tables optimized for analysis:

1. **papers** — Core publication metadata
2. **authorships** — Author-paper relationships
3. **citations** — Citation network edges
4. **coauthorships** — Coauthor collaboration network edges

All tables are stored as Parquet files (or CSV if Parquet is unavailable) partitioned into multiple part files for scalability.

---

## Table: `papers`

**Description:** Contains one record per publication with metadata and derived metrics.

**Primary Key:** `id`

**Storage Location:** `data/parquet/papers/part-*.parquet`

### Columns

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `id` | string | No | Unique paper identifier from DBLP dataset |
| `title` | string | Yes | Full title of the publication |
| `venue` | string | Yes | Publication venue (conference, journal, workshop, etc.) |
| `year` | integer | Yes | Publication year (validated range: 1900-2030) |
| `n_citation` | integer | Yes | Citation count from original DBLP data (may be outdated; prefer computing from `citations` table) |
| `abstract` | string | Yes | Full abstract text (if available) |
| `abstract_len` | integer | Yes | Character length of abstract (NULL if no abstract) |
| `ref_count` | integer | No | Number of references cited by this paper (derived from `references` array) |
| `author_count` | integer | No | Number of authors on this paper |

### Indexes and Optimization

- **Primary lookups:** Index on `id` for fast joins
- **Temporal analysis:** Index on `year` for time-series queries
- **Venue analysis:** Index on `venue` for conference/journal-specific queries
- **Impact analysis:** Index on `n_citation` for highly-cited paper retrieval

### Notes

- Papers without abstracts will have `NULL` in both `abstract` and `abstract_len`
- `n_citation` is from the source data and may not reflect the actual citation count in the dataset; compute in-degree from `citations` table for accurate counts
- `ref_count` counts only references that exist in the `references` field; some older papers may have incomplete reference lists

---

## Table: `authorships`

**Description:** Many-to-many relationship between papers and authors. Each row represents one author's contribution to one paper.

**Primary Key:** None (relationship table)

**Foreign Keys:** 
- `paper_id` → `papers.id`

**Storage Location:** `data/parquet/authorships/part-*.parquet`

### Columns

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `paper_id` | string | No | Foreign key to `papers.id` |
| `author_name` | string | Yes | Author name as it appears in the publication (raw, unnormalized) |
| `author_position` | integer | No | Zero-based position of author in author list (0 = first author) |
| `author_norm` | string | Yes | Normalized author name for deduplication and analysis |

### Normalization Process (`author_norm`)

The `author_norm` field applies the following transformations to `author_name`:

1. Strip leading/trailing whitespace
2. Convert to lowercase
3. Collapse multiple spaces to single space
4. Remove Unicode accents using NFKD normalization
5. Remove combining characters

**Example:**
- Raw: `"José García-López  "`
- Normalized: `"jose garcia-lopez"`

### Indexes and Optimization

- **Paper lookups:** Index on `paper_id` for author retrieval per paper
- **Author lookups:** Index on `author_norm` for publication lists per author
- **Author position analysis:** Compound index on (`author_norm`, `author_position`) for first-author queries

### Notes

- Author disambiguation is NOT performed beyond normalization; authors with similar names may be conflated
- Multiple spellings of the same author will appear as different `author_norm` values
- Some papers may have missing or NULL author names in the source data

---

## Table: `citations`

**Description:** Directed citation network. Each row represents one paper citing another.

**Primary Key:** None (edge table; may have duplicate edges in rare cases)

**Foreign Keys:**
- `src_id` → `papers.id` (citing paper)
- `dst_id` → May or may not exist in `papers.id` (cited paper could be external to dataset)

**Storage Location:** `data/parquet/citations/part-*.parquet`

### Columns

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `src_id` | string | No | Paper ID of the citing paper (source of citation edge) |
| `dst_id` | string | No | Paper ID of the cited paper (target of citation edge) |
| `src_year` | integer | Yes | Publication year of citing paper (denormalized for temporal analysis) |
| `src_venue` | string | Yes | Venue of citing paper (denormalized for cross-venue citation analysis) |

### Graph Properties

- **Directed:** Yes (A → B means A cites B)
- **Weighted:** No (use `GROUP BY` on (`src_id`, `dst_id`) to compute edge weights if needed)
- **Self-loops:** Possible but rare (paper citing itself)
- **External nodes:** `dst_id` may reference papers not in the `papers` table

### Indexes and Optimization

- **Out-degree (papers citing):** Index on `src_id`
- **In-degree (papers cited):** Index on `dst_id`
- **Temporal citation patterns:** Index on `src_year`
- **Cross-venue citations:** Compound index on (`src_venue`, `dst_id`)

### Derived Metrics

```sql
-- In-degree (citation count) for each paper
SELECT dst_id, COUNT(*) as citation_count
FROM citations
GROUP BY dst_id

-- Out-degree (reference count) for each paper
SELECT src_id, COUNT(*) as reference_count
FROM citations
GROUP BY src_id

-- Self-citations
SELECT COUNT(*) FROM citations WHERE src_id = dst_id
```

### Notes

- Not all cited papers (`dst_id`) appear in the `papers` table (external references)
- Citation year lags can be computed as `citing_year - cited_year` (requires join with `papers`)
- Self-citations (where `src_id = dst_id`) should be filtered for most network analyses

---

## Table: `coauthorships`

**Description:** Undirected collaboration network. Each row represents one coauthorship instance on one paper.

**Primary Key:** None (multiple collaborations between same pair expected)

**Storage Location:** `data/parquet/coauthorships/part-*.parquet`

### Columns

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `author1_norm` | string | No | Normalized name of first author in the pair (lexicographically smaller) |
| `author2_norm` | string | No | Normalized name of second author in the pair (lexicographically larger) |
| `paper_id` | string | No | Paper on which this collaboration occurred |
| `year` | integer | Yes | Publication year of the collaborative paper |
| `venue` | string | Yes | Venue where the collaborative paper was published |

### Edge Construction

- **Undirected:** Author pairs are sorted alphabetically to ensure consistency
  - If authors are [A, B, C], edges created are: (A,B), (A,C), (B,C)
  - Each edge is stored with `author1_norm < author2_norm`
- **Multi-edges:** The same author pair may appear multiple times (one row per joint paper)

### Indexes and Optimization

- **Author neighborhood:** Compound index on (`author1_norm`, `author2_norm`)
- **Reverse lookups:** Compound index on (`author2_norm`, `author1_norm`)
- **Temporal collaboration:** Index on `year`
- **Paper collaboration details:** Index on `paper_id`

### Derived Metrics

```sql
-- Collaboration frequency (number of joint papers)
SELECT author1_norm, author2_norm, COUNT(*) as collaboration_count
FROM coauthorships
GROUP BY author1_norm, author2_norm

-- Author degree (number of unique collaborators)
SELECT author, COUNT(DISTINCT collaborator) as num_collaborators
FROM (
  SELECT author1_norm as author, author2_norm as collaborator FROM coauthorships
  UNION ALL
  SELECT author2_norm as author, author1_norm as collaborator FROM coauthorships
) t
GROUP BY author

-- Collaboration timeline for an author
SELECT year, COUNT(DISTINCT paper_id) as papers_with_coauthors
FROM coauthorships
WHERE author1_norm = 'target_author' OR author2_norm = 'target_author'
GROUP BY year
ORDER BY year
```

### Notes

- Edges are deduplicated by author pair but NOT by paper; same pair appears once per joint publication
- Single-author papers generate no coauthorship edges
- Author normalization issues (e.g., name variations) can affect collaboration network accuracy

---

## Data Quality and Validation

### Referential Integrity

| Check | Expected | Notes |
|-------|----------|-------|
| `authorships.paper_id` → `papers.id` | 100% match | All authorships should reference valid papers |
| `citations.src_id` → `papers.id` | 100% match | All citing papers should be in dataset |
| `citations.dst_id` → `papers.id` | Partial match | Cited papers may be external to dataset |
| `coauthorships.paper_id` → `papers.id` | 100% match | All coauthorships should reference valid papers |

### Missing Data Patterns

**High missingness expected:**
- `papers.abstract`: ~40-60% missing (older papers lack abstracts)
- `papers.venue`: ~5-15% missing (some publications have unspecified venues)

**Low missingness expected:**
- `papers.title`: <1% (core field)
- `papers.year`: <5% (mostly present)
- `authorships.author_name`: <1% (rare)

### Data Anomalies to Check

1. **Duplicate paper IDs** in `papers` table
2. **Invalid years** (e.g., year < 1900 or year > 2030)
3. **Self-citations** in `citations` table
4. **Orphaned records** (authorships/citations referencing non-existent papers)
5. **Author normalization collisions** (different authors with identical normalized names)

---

## Usage Examples

### Join Papers with Authors

```sql
SELECT p.id, p.title, a.author_name, a.author_position
FROM papers p
JOIN authorships a ON p.id = a.paper_id
WHERE p.year = 2020
ORDER BY p.id, a.author_position
```

### Compute Citation Impact by Venue

```sql
SELECT p.venue, COUNT(*) as paper_count, AVG(cite_count) as avg_citations
FROM papers p
LEFT JOIN (
  SELECT dst_id, COUNT(*) as cite_count
  FROM citations
  GROUP BY dst_id
) c ON p.id = c.dst_id
WHERE p.venue IS NOT NULL
GROUP BY p.venue
HAVING paper_count >= 100
ORDER BY avg_citations DESC
```

### Find Frequent Collaborators

```sql
SELECT author1_norm, author2_norm, COUNT(*) as num_papers,
       MIN(year) as first_collab, MAX(year) as last_collab
FROM coauthorships
GROUP BY author1_norm, author2_norm
HAVING num_papers >= 5
ORDER BY num_papers DESC
```

### Temporal Analysis of Research Output

```sql
SELECT year, 
       COUNT(*) as papers,
       AVG(author_count) as avg_authors,
       AVG(ref_count) as avg_references
FROM papers
WHERE year BETWEEN 2000 AND 2023
GROUP BY year
ORDER BY year
```

---

## Storage and Performance Guidelines

### File Formats

- **Parquet** (preferred): Columnar storage, efficient compression, fast analytics
  - Typical compression ratio: 5-10x vs CSV
  - Supports predicate pushdown for faster queries
  
- **CSV** (fallback): Row-oriented, human-readable, slower for analytics
  - Used when `pyarrow` is not installed

### Partitioning Strategy

- Data is partitioned into multiple part files per table (default: 50,000 records per part)
- Benefits:
  - Parallel reading/writing
  - Memory-efficient streaming
  - Easier incremental updates

### Query Optimization

**For Pandas:**
```python
# Read only needed columns
df = pd.read_parquet('papers/part-*.parquet', columns=['id', 'year', 'title'])

# Read with filters (Parquet only)
df = pd.read_parquet('papers/', filters=[('year', '>=', 2010)])
```

**For DuckDB (recommended for large datasets):**
```python
import duckdb
conn = duckdb.connect()
result = conn.execute("""
    SELECT year, COUNT(*) as count
    FROM read_parquet('data/parquet/papers/part-*.parquet')
    WHERE year >= 2010
    GROUP BY year
""").fetchdf()
```

**For Apache Spark:**
```python
df = spark.read.parquet('data/parquet/papers/')
df.filter(df.year >= 2010).groupBy('year').count().show()
```

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-01 | Initial data dictionary with all four tables |

---

## Contact and Contributions

For questions about this data dictionary or to report issues with the ETL pipeline, please refer to the project README or open an issue in the repository.
