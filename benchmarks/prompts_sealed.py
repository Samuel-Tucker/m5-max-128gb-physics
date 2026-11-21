#!/usr/bin/env python3
"""
Sealed 12-Prompt Benchmark Suite across 3 Distinct Workload Categories (4 each).

Categories are intentionally separated so copy-heavy workloads NEVER disguise
general high-entropy prose:
  1. code_edit  (4 prompts): Realistic Python/Metal code editing & refactoring.
  2. extraction (4 prompts): Realistic document extraction / structured JSON / RAG.
  3. free_prose (4 prompts): Hard zero-copy math proofs, creative prose, first-principles reasoning.
"""

from __future__ import annotations

from typing import Any, Dict, List


SEALED_PROMPTS: List[Dict[str, str]] = [
    # =========================================================================
    # Category 1: Code Editing / Refactoring (4 prompts)
    # =========================================================================
    {
        "id": "code_edit_1",
        "category": "code_edit",
        "title": "Python TTL-aware Thread-Safe LRU Cache Refactor",
        "prompt": """Refactor the Python class below to add:
1. Optional `ttl_seconds: Optional[float] = None` per entry (defaulting to `self.default_ttl`).
2. Thread safety using `threading.RLock`.
3. A `purge_expired() -> int` method that removes expired entries and returns the count removed.
4. Keep all existing method names, docstrings, and `stats()` counters intact (add `"expired"` to `stats()`).

Output the complete updated Python code only.

```python
from collections import OrderedDict
from typing import Any, Dict, Optional


class SimpleLRUCache:
    \"\"\"Bounded in-memory LRU cache with hit/miss tracking.\"\"\"

    def __init__(self, max_items: int = 256) -> None:
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        self.max_items = max_items
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._store:
            self._misses += 1
            return default
        self._store.move_to_end(key)
        self._hits += 1
        return self._store[key]

    def put(self, key: str, value: Any) -> None:
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = value
            return
        if len(self._store) >= self.max_items:
            self._store.popitem(last=False)
            self._evictions += 1
        self._store[key] = value

    def delete(self, key: str) -> bool:
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> None:
        self._store.clear()

    def stats(self) -> Dict[str, int]:
        return {
            "size": len(self._store),
            "max_items": self.max_items,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
        }
```""",
    },
    {
        "id": "code_edit_2",
        "category": "code_edit",
        "title": "Metal Shading Language Fused Q4 MatVec + SiLU Refactor",
        "prompt": """Update the Metal Shading Language (MSL) kernel below to:
1. Add a second quantized weight matrix (`up_weights`, `up_scales`, `up_biases`) at buffers 6, 7, 8 to compute fused SwiGLU: `silu(gate_acc) * up_acc`.
2. Add an explicit bounds check `if (out_row >= N) return;`.
3. Accumulate both `gate_acc` and `up_acc` in `float` using `simd_sum`.
4. Keep the exact buffer binding conventions and group_size=64 unpacking structure.

Output the complete updated Metal kernel only.

```metal
#include <metal_stdlib>
using namespace metal;

constant uint GROUP_SIZE = 64;
constant uint PACK_FACTOR = 8;

kernel void q4_affine_matvec(
    device const half* x [[buffer(0)]],
    device const uint32_t* gate_weights [[buffer(1)]],
    device const half* gate_scales [[buffer(2)]],
    device const half* gate_biases [[buffer(3)]],
    device half* y [[buffer(4)]],
    constant uint2& dims [[buffer(5)]],
    uint tid [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]])
{
    const uint N = dims.x;
    const uint K = dims.y;
    const uint out_row = tg_id * 4 + simd_id;
    const uint groups_per_row = K / GROUP_SIZE;
    const uint words_per_row = K / PACK_FACTOR;

    device const uint32_t* w_row = gate_weights + out_row * words_per_row;
    device const half* s_row = gate_scales + out_row * groups_per_row;
    device const half* b_row = gate_biases + out_row * groups_per_row;

    float gate_acc = 0.0f;
    for (uint g = tid; g < groups_per_row; g += 32) {
        const float scale = float(s_row[g]);
        const float bias = float(b_row[g]);
        const uint k_base = g * GROUP_SIZE;
        const uint w_base = g * (GROUP_SIZE / PACK_FACTOR);
        for (uint p = 0; p < GROUP_SIZE / PACK_FACTOR; ++p) {
            uint32_t packed = w_row[w_base + p];
            for (uint i = 0; i < PACK_FACTOR; ++i) {
                float qval = float((packed >> (4 * i)) & 0xFu);
                float w = qval * scale + bias;
                gate_acc += w * float(x[k_base + p * PACK_FACTOR + i]);
            }
        }
    }

    gate_acc = simd_sum(gate_acc);
    if (tid == 0) {
        y[out_row] = half(gate_acc);
    }
}
```""",
    },
    {
        "id": "code_edit_3",
        "category": "code_edit",
        "title": "UK VAT & Penny-Exact GBP Order Repository Edit",
        "prompt": """Edit the Python `OrderRepository` code below to:
1. Replace float `net_gbp` with integer `net_pence: int` and `vat_rate_bps: int = 2000` (20.00% standard UK VAT, supporting 0 bps zero-rated PPE exemptions).
2. Add computed properties `vat_pence` (half-up integer rounding) and `gross_pence` on `OrderRecord`.
3. Add `update_status(self, order_id: str, new_status: str) -> bool` validating against `VALID_STATUSES = {"pending", "picked", "dispatched", "cancelled"}`.
4. Keep the SQLite schema initialization, `insert_order`, and `list_by_customer` intact and updated for the new columns.

Output the complete updated Python module only.

```python
from dataclasses import dataclass
import sqlite3
from typing import List, Optional


@dataclass
class OrderRecord:
    order_id: str
    customer_code: str
    sku: str
    quantity: int
    net_gbp: float
    status: str = "pending"


class OrderRepository:
    def __init__(self, db_path: str) -> None:
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            \"\"\"
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                customer_code TEXT NOT NULL,
                sku TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                net_gbp REAL NOT NULL,
                status TEXT NOT NULL
            )
            \"\"\"
        )
        self.conn.commit()

    def insert_order(self, rec: OrderRecord) -> None:
        self.conn.execute(
            \"\"\"
            INSERT INTO orders (order_id, customer_code, sku, quantity, net_gbp, status)
            VALUES (?, ?, ?, ?, ?, ?)
            \"\"\",
            (rec.order_id, rec.customer_code, rec.sku, rec.quantity, rec.net_gbp, rec.status),
        )
        self.conn.commit()

    def list_by_customer(self, customer_code: str) -> List[OrderRecord]:
        rows = self.conn.execute(
            "SELECT order_id, customer_code, sku, quantity, net_gbp, status FROM orders WHERE customer_code = ?",
            (customer_code,),
        ).fetchall()
        return [OrderRecord(**dict(r)) for r in rows]
```""",
    },
    {
        "id": "code_edit_4",
        "category": "code_edit",
        "title": "Hybrid SSM Checkpoint Manager Refactor",
        "prompt": """Refactor the Python `HybridStateCheckpoint` class below to:
1. Support optional per-layer `float32` dtype enforcement on recurrent states (`assert_fp32: bool = True`).
2. Add a `diff_summary(self, other: "HybridStateCheckpoint") -> Dict[str, float]` method computing `max_conv_diff`, `max_state_diff`, and `max_kv_offset_diff`.
3. Add a `memory_mib(self) -> float` helper summing `nbytes` across stored arrays.
4. Preserve all existing attributes, docstrings, and `save` / `restore` semantics.

Output the complete updated Python code only.

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import mlx.core as mx


@dataclass
class HybridStateCheckpoint:
    \"\"\"Lightweight snapshot of Gated DeltaNet and full-attention KV positions.\"\"\"

    linear_states: Dict[int, Tuple[Optional[mx.array], Optional[mx.array]]] = field(default_factory=dict)
    kv_offsets: Dict[int, int] = field(default_factory=dict)
    token_position: int = 0

    @classmethod
    def capture(cls, cache: List, linear_indices: List[int], kv_indices: List[int], token_position: int) -> "HybridStateCheckpoint":
        lin = {}
        for idx in linear_indices:
            lin[idx] = (cache[idx][0], cache[idx][1])
        kv = {}
        for idx in kv_indices:
            kv[idx] = int(cache[idx].offset)
        return cls(linear_states=lin, kv_offsets=kv, token_position=token_position)

    def restore(self, cache: List) -> None:
        for idx, (conv_s, rec_s) in self.linear_states.items():
            cache[idx][0] = conv_s
            cache[idx][1] = rec_s
        for idx, off in self.kv_offsets.items():
            cache[idx].offset = off
```""",
    },

    # =========================================================================
    # Category 2: Document Extraction / Structured JSON / Summarization (4 prompts)
    # =========================================================================
    {
        "id": "extraction_1",
        "category": "extraction",
        "title": "PPE & Welding Supplier Delivery Note to Structured JSON",
        "prompt": """Extract all structured fields from the supplier delivery note below into a single valid JSON object with keys: `supplier`, `delivery_note_no`, `dispatch_date`, `warehouse`, `currency`, `vat_rate_pct`, `line_items` (array of `{sku, description, bin_location, quantity, unit_price_gbp, line_net_gbp}`), `subtotal_net_gbp`, `vat_gbp`, `total_gross_gbp`, and `special_instructions`. Output JSON only.

---
SUPPLIER DELIVERY NOTE — NORTHUMBRIA INDUSTRIAL SAFETY LTD
Delivery Note No: NIS-2026-08419
Dispatch Date: 18 November 2026
Receiving Warehouse: Pricing Logic Logistics — Unit 14, Tyne Tunnel Trading Estate, North Shields, NE29 7XJ
Currency: GBP | UK Standard VAT: 20.0%

Line Items Dispatched:
1. SKU: PAPR-3M-TR602E | 3M Versaflo TR-602E Powered Air Turbo Unit (Body Only) | Bin: A-04-02 | Qty: 6 | Unit Net: £412.50 | Line Net: £2,475.00
2. SKU: FILT-TR6310E | 3M Versaflo A2P Particulate & Organic Vapour Filter Cartridge (Pack of 5) | Bin: A-04-09 | Qty: 18 | Unit Net: £64.00 | Line Net: £1,152.00
3. SKU: HOOD-S433L | 3M Versaflo S-433L Integrated Head Suspension Hood (M/L) | Bin: B-02-11 | Qty: 25 | Unit Net: £38.40 | Line Net: £960.00
4. SKU: WELD-ESAB-A60 | ESAB Sentinel A60 Auto-Darkening Welding Helmet (DIN 3/5-13) | Bin: C-01-04 | Qty: 10 | Unit Net: £249.00 | Line Net: £2,490.00
5. SKU: BATT-TR632 | 3M Versaflo TR-632 High Capacity Lithium-Ion Battery Pack | Bin: A-04-05 | Qty: 12 | Unit Net: £158.00 | Line Net: £1,896.00
6. SKU: LENS-A60-CLR | ESAB Sentinel A60 Clear Outer Cover Lens (Box of 10) | Bin: C-01-08 | Qty: 30 | Unit Net: £19.50 | Line Net: £585.00

Totals:
Subtotal Net: £9,558.00
UK VAT (20%): £1,911.60
Total Gross Payable: £11,469.60

Special Instructions: Inspect lithium-ion battery cartons (BATT-TR632) for UN3480 labelling before shelving in Cage A-04. Notify warehouse supervisor Mark Henderson if any TR-602E serial seals are broken on arrival.
---""",
    },
    {
        "id": "extraction_2",
        "category": "extraction",
        "title": "Apple Silicon GPU Memory Wall Incident Report Extraction",
        "prompt": """Extract a structured JSON technical summary from the benchmark report below with keys: `system`, `models_tested` (list of objects with `name`, `active_bytes_gb`, `b1_latency_ms`, `achieved_bw_gbs`, `bottleneck`), `dispatch_cliff` (`library_version`, `file_and_line`, `m64_ms`, `m65_ms`, `speedup_pct`, `root_cause`), `architectural_traps` (list of `{id, summary, mitigation}`), and `hardware_discoveries` (list of strings). Output JSON only.

---
ENGINEERING BENCHMARK REPORT — M5 MAX INFERENCE PROFILING (20 NOV 2026)
System: Apple M5 Max (40 GPU cores, 128 GB unified LPDDR5X, measured R+W copy 503 GB/s, FP16 NAX 59.9 TFLOP/s).

Models Profiled:
- mlx-community/Qwen3.5-9B-4bit: Active decode bytes 4.47 GB (excluding vision tower), B=1 decode latency 10.66 ms (~93.8 tok/s), achieved DRAM bandwidth 419–525 GB/s. Bottleneck: already near physical LPDDR5X memory wall; 97% of FP16 NAX compute sits idle at B=1.
- mlx-community/Qwen3.6-35B-A3B-4bit: Active decode bytes ~1.67 GB (8 of 256 experts active per layer), B=1 decode latency ~9.1 ms (~110 tok/s), achieved bandwidth ~185 GB/s (37% of memory wall). Bottleneck: occupancy-starved B=1 switch_mlp gather/dispatch tax.

Critical Dispatch Cliff Found:
In MLX 0.31.2 (`mlx/backend/metal/quantized.cpp:1810`), quantized matrix multiplication for N=4096 at M=8..64 routes to legacy non-NAX `qmm_t_splitk` instead of Metal 4 NAX `qmm_nax`. Consequently, a 32-layer Q4 4096x4096 matmul at M=64 takes 5.45 ms, whereas padding 1 zero row to M=65 drops latency to 3.90 ms (28.4% faster!).

Architectural Traps Identified:
1. Gated DeltaNet Hybrid Recurrence: Qwen3.5-9B (24/32 layers) and Qwen3.6-35B-A3B (30/40 layers) store recurrent state matrices in `ArraysCache(size=2)`, which stock `mlx_lm` cannot trim during speculative decoding. Mitigation: zero-copy checkpointing + zero-matmul recurrent factor replay.
2. MoE Expert Explosion & Serial 2B Drafter Cap: Wide speculation (K=16..64) on 35B-A3B expands unique active experts from 8 to 84..144 per layer, destroying bandwidth savings. Meanwhile a serial Qwen3.5-2B-4bit drafter (3.85 ms/tok) caps 9B at ~1.55x even at 100% acceptance. Mitigation: zero-byte CPU suffix/lookahead proposers + shallow K=3..4 on MoE.

Unexposed Hardware Discoveries:
- Metal 4 SDK `MPPTensorOpsMatMul2d.h` supports direct hardware `half x uint4b_format -> float` without threadgroup dequant buffers.
- Hardware INT8xINT8 NAX achieves 84.3 TOPS and INT8xINT4 achieves 78.7 TOPS.
- Qwen3.5-9B MLP `gate_proj` outputs are negative on 73% to 87% of channels.
---""",
    },
    {
        "id": "extraction_3",
        "category": "extraction",
        "title": "UK B2B SLA Contract Clause Extraction",
        "prompt": """Extract the key commercial and legal terms from the contract excerpt below into valid JSON with keys: `provider`, `client`, `effective_date`, `initial_term_months`, `annual_fee_gbp_ex_vat`, `payment_terms_days`, `uptime_target_pct`, `response_slas` (list of `{severity, response_time, resolution_target}`), `service_credits` (list of `{availability_band, credit_pct}`), `governing_law`, and `notice_period_days`. Output JSON only.

---
MANAGED SOFTWARE SUPPORT & SLA SCHEDULE
Agreement between Pricing Logic Ltd (Company No. 11842903, registered in England and Wales, "Provider") and Tyne Valley Distribution Holdings Ltd ("Client").
Effective Date: 1 December 2026. Initial Term: 24 months, renewing annually thereafter unless either party gives at least 90 days' written notice prior to the renewal date.

Fees & Invoicing:
The Annual Support Fee is £18,600.00 excluding UK VAT, invoiced quarterly in advance (£4,650.00 + VAT per quarter). Invoices are payable within 30 days of invoice date by BACS transfer in Pounds Sterling (GBP).

Service Availability & Response SLAs:
Provider shall use commercially reasonable endeavours to achieve 99.90% monthly platform availability (excluding scheduled Sunday maintenance between 02:00 and 04:00 GMT).
- Severity 1 (Critical Outage — complete loss of order processing): Initial response within 30 minutes; target resolution within 4 hours.
- Severity 2 (Major Degradation — core pricing sync delayed >15 minutes): Initial response within 2 hours; target resolution within 1 business day.
- Severity 3 (Minor Defect — non-blocking UI or export issue): Initial response within 1 business day; target resolution within 5 business days.

Service Credit Schedule (measured per calendar month):
- Monthly availability >= 99.50% and < 99.90%: 5% credit on monthly fee.
- Monthly availability >= 99.00% and < 99.50%: 10% credit on monthly fee.
- Monthly availability < 99.00%: 20% credit on monthly fee (maximum monthly credit cap).

Governing Law: This Agreement and any dispute arising out of it shall be governed by and construed in accordance with the laws of England and Wales.
---""",
    },
    {
        "id": "extraction_4",
        "category": "extraction",
        "title": "Trading Card Auction Catalogue to Structured JSON",
        "prompt": """Parse the trading card auction lot list below into a JSON object with keys `auction_house`, `sale_date`, `buyers_premium_pct`, `lots` (array of `{lot_no, franchise, card_name, set_name, year, card_number, grading_company, grade, cert_number, hammer_price_gbp, total_with_premium_gbp}`), and `highest_lot_no`. Output JSON only.

---
SHENTANCARD PREMIER GRADED AUCTION — LOT RESULTS SUMMARY
Auction House: Heritage UK Collectibles (London)
Sale Date: 14 November 2026 | Buyer's Premium: 15% (applied to hammer price)

Lot 101: Pokémon — Charizard Holo Unlimited | Base Set (1999) #4/102 | PSA 9 Mint | Cert #84920144 | Hammer: £1,420.00 | Total incl. 15% BP: £1,633.00
Lot 102: Pokémon — Umbreon VMAX Alternate Art Secret | Evolving Skies (2021) #215/203 | PSA 10 Gem Mint | Cert #91204588 | Hammer: £980.00 | Total incl. 15% BP: £1,127.00
Lot 103: Topps Chrome Formula 1 — Lewis Hamilton Gold Refractor /50 | Topps Chrome F1 (2020) #1 | BGS 9.5 Gem Mint | Cert #0015842910 | Hammer: £2,650.00 | Total incl. 15% BP: £3,047.50
Lot 104: Panini Prizm Premier League — Erling Haaland Silver Prizm | Prizm EPL (2022) #146 | PSA 10 Gem Mint | Cert #77419023 | Hammer: £310.00 | Total incl. 15% BP: £356.50
Lot 105: Pokémon — Lugia Neo Genesis 1st Edition Holo | Neo Genesis (2000) #9/111 | CGC 8.5 NM/Mint+ | Cert #4190283004 | Hammer: £1,890.00 | Total incl. 15% BP: £2,173.50
Lot 106: Topps Merlin Chrome UEFA — Jude Bellingham Auto /99 | Merlin Chrome (2021) #BA-JB | PSA 9 Mint | Cert #88301927 | Hammer: £740.00 | Total incl. 15% BP: £851.00
---""",
    },

    # =========================================================================
    # Category 3: Free Prose / Math / First-Principles Reasoning (4 prompts)
    # =========================================================================
    {
        "id": "free_prose_1",
        "category": "free_prose",
        "title": "Roofline-Optimal Speculative Decoding Block Length Derivation",
        "prompt": (
            "Derive from first principles the optimal draft block length K* for lossless "
            "speculative decoding with a zero-cost drafter (0 ms draft latency) under a "
            "memory-bandwidth-plus-compute roofline verification model: "
            "T_verify(K) = T_0 + c * max(0, K + 1 - M_ridge), where T_0 is the DRAM-bound "
            "weight-streaming floor, M_ridge is the hardware compute-to-bandwidth ridge point, "
            "and each drafted token is accepted independently with probability alpha in (0, 1). "
            "Write out the exact expected committed tokens E[N(K)], compute the throughput "
            "objective, find the break-even acceptance rate alpha_min, and show why K* <= M_ridge - 1 "
            "is not always optimal when alpha is close to 1."
        ),
    },
    {
        "id": "free_prose_2",
        "category": "free_prose",
        "title": "Original Literary Prose: Lighthouse Acoustic Resonance",
        "prompt": (
            "Write an original, atmospheric literary passage (continuous narrative prose, no "
            "bullet points or headings) about a solitary lighthouse keeper stationed on an "
            "isolated granite skerry off the west coast of Scotland during a November gale. "
            "While inspecting the nineteenth-century brass diaphone foghorn at midnight, she "
            "notices that the iron stairwell beneath the lantern room has begun humming an "
            "unexpected fifth below the horn's fundamental pitch. Focus on concrete sensory "
            "details—salt spray, oil lamps, pressure gauges, resonance through boot soles—and "
            "her methodical curiosity as she investigates the cause."
        ),
    },
    {
        "id": "free_prose_3",
        "category": "free_prose",
        "title": "Comparative Neurobiology: Distributed Octopus Nervous System",
        "prompt": (
            "Analyze from first principles why cephalopods such as Octopus vulgaris evolved a "
            "radically decentralized nervous system—with roughly two-thirds of their neurons "
            "located in eight brachial nerve cords rather than inside the central brain—whereas "
            "vertebrates centralized motor planning in a layered cortex and cerebellum. Compare "
            "both architectures quantitatively in terms of axonal conduction latency, sensory "
            "bandwidth from flexible hydrostatic arms with infinite degrees of freedom, metabolic "
            "wiring cost, and failure modes. Conclude with two concrete design lessons for "
            "distributed AI inference hardware."
        ),
    },
    {
        "id": "free_prose_4",
        "category": "free_prose",
        "title": "Rigorous Number Theory: Divergence of Sum of Reciprocals of Primes",
        "prompt": (
            "Give a complete, self-contained proof of Paul Erdos's elementary argument that the "
            "series sum_{p prime} 1/p diverges, carefully bounding the number of integers up to N "
            "that are 'smooth' (divisible only by the first k primes) versus 'rough' (divisible by "
            "at least one prime larger than p_k), and explicitly showing how choosing N sufficiently "
            "large produces a contradiction. Afterward, explain intuitively—without circularly "
            "assuming the Prime Number Theorem—why this divergence implies primes cannot thin out "
            "as fast as squares or cubes."
        ),
    },
]


def get_prompts(prompts_per_cat: int = 4) -> List[Dict[str, str]]:
    """Return balanced subset of sealed prompts (prompts_per_cat per category)."""
    cats = ["code_edit", "extraction", "free_prose"]
    selected: List[Dict[str, str]] = []
    for cat in cats:
        cat_items = [p for p in SEALED_PROMPTS if p["category"] == cat]
        selected.extend(cat_items[:prompts_per_cat])
    return selected


def format_prompt_tokens(tokenizer: Any, prompt_text: str) -> List[int]:
    """Format prompt using Qwen3.5/3.6 chat template with thinking disabled."""
    messages = [{"role": "user", "content": prompt_text}]
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return list(tokenizer.encode(rendered))
