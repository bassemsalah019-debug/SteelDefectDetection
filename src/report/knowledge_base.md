# Steel Surface Defect Knowledge Base (NEU-DET, 6 classes)

Human-readable companion to `knowledge_base.json` (the machine-read grounding for
the LLM report). The JSON is the source of truth; this file mirrors the English
fields for review. The LLM is instructed to explain **only** detected defects and
to use **only** this content — it must not invent metallurgy.

| Class | Definition | Likely root cause | Visual signature | Severity |
|---|---|---|---|---|
| **Crazing** | Network of fine interconnected hairline cracks. | Thermal/residual stress from uneven cooling, excessive hardening, rapid quench. | Web-like mesh of shallow cracks; diffuse, low contrast. | moderate–high (stress concentrators; can propagate) |
| **Inclusion** | Non-metallic particles (oxides, sulfides, slag) embedded in the surface. | Impurities/deoxidation products or refractory erosion not floated out before solidification. | Discrete dark spots, streaks, stringers. | moderate–high (fatigue-crack initiation) |
| **Patches** | Irregular discoloration / uneven texture. | Uneven pickling, localized oxidation, residual scale/coating non-uniformity. | Irregular bright/dark areas, diffuse edges. | low–moderate (often cosmetic; may mask corrosion) |
| **Pitted surface** | Clusters of small cavities/pits. | Localized corrosion, trapped gas, over-pickling, micro-damage. | Dotted depressions / speckled clusters. | moderate (corrosion + fatigue initiation) |
| **Rolled-in scale** | Oxide scale pressed into the surface during hot rolling. | Scale not removed before rolling, then rolled in. | Dark elongated/patchy scale along rolling direction. | moderate (surface integrity, coating adhesion) |
| **Scratches** | Linear grooves / score marks. | Mechanical contact in handling, rollers/guides, hard particles. | Straight/curved high-contrast directional lines. | low–moderate (deep ones are stress raisers) |

Each class also carries a **recommended_action** and an Arabic (`ar`) translation
of every field in the JSON, used for the bilingual report and the deterministic
KB-only fallback.
