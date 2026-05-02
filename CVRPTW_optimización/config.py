"""
config.py  —  Central configuration for the school bus routing pipeline.

Project layout:
    project/
    ├── config.py
    ├── generate_data.py
    ├── optimize_routes.py
    ├── visualize_results.py
    ├── run_all.py
    └── data/
        ├── comunas_medellin.geojson
        ├── barrios_medellin.geojson
        ├── valle_aburra_urbano.geojson
        └── poblation_barrios_medellin_2026.csv
"""

# ── Reproducibility ───────────────────────────────────────────────────────
SEED = 42

# ── Road network bounding box (full Valle de Aburrá) ─────────────────────
BBOX = dict(north=6.435, south=6.030, east=-75.330, west=-75.680)

# ── School / depot ────────────────────────────────────────────────────────
SCHOOL_LON  = -75.58683269546758
SCHOOL_LAT  =  6.243283526198216
SCHOOL_NAME = "UPB"

# ── Scenario ──────────────────────────────────────────────────────────────
N_CHILDREN   = 200   # total children across the entire Valle
BUS_CAPACITY = 20    # max children per bus

# ── DANE 2026 population estimates for non-Medellín municipalities ────────
MUNICIPALITY_POPULATION = {
    "Bello":        566_000,
    "Itagüí":       280_000,
    "Envigado":     240_000,
    "Sabaneta":     110_000,
    "La Estrella":   70_000,
}

# ── 6 Routing zones ───────────────────────────────────────────────────────
# Children are SAMPLED by individual comuna/municipality (generate_data.py).
# Routes are OPTIMISED by these 6 broader zones (optimize_routes.py).
# Each zone gets its own CVRPTW and its own set of buses.
ZONE_DEFINITIONS = [
    {
        "zone_name": "Norte",
        "color":     "#3A86FF",
        "members": [
            "Bello",
        ],
    },
    {
        "zone_name": "Nororiental",
        "color":     "#FF006E",
        "members": [
            "Comuna 1 - Popular",
            "Comuna 2 - Santa Cruz",
            "Comuna 3 - Manrique",
        ],
    },
    {
        "zone_name": "Noroccidental",
        "color":     "#FB5607",
        "members": [
            "Comuna 4 - Aranjuez",
            "Comuna 5 - Castilla",
            "Comuna 6 - Doce de Octubre",
            "Comuna 7 - Robledo",
        ],
    },
    {
        "zone_name": "Centro",
        "color":     "#FFBE0B",
        "members": [
            "Comuna 10 - La Candelaria",
            "Comuna 11 - Laureles-Estadio",
            "Comuna 12 - La América",
            "Comuna 13 - San Javier",
        ],
    },
    {
        "zone_name": "Suroriental",
        "color":     "#8338EC",
        "members": [
            "Comuna 8 - Villa Hermosa",
            "Comuna 9 - Buenos Aires",
            "Comuna 14 - El Poblado",
            "Envigado",
        ],
    },
    {
        "zone_name": "Sur",
        "color":     "#06D6A0",
        "members": [
            "Comuna 15 - Guayabal",
            "Comuna 16 - Belén",
            "Itagüí",
            "Sabaneta",
            "La Estrella",
        ],
    },
]

# ── Time-window / routing model ───────────────────────────────────────────
MAX_ROUTE_MINUTES  = 75    # hard upper bound per route
BOARDING_SECONDS   = 90    # avg boarding time added at each stop
SPEED_FALLBACK_KPH = 30    # used when OSM maxspeed tag is absent

# ── Multi-objective weights ───────────────────────────────────────────────
# OR-Tools objective = total_travel_time  +  SPAN_COEFF × route_time_span
# Higher SPAN_COEFF → more balanced routes; lower → shorter total distance.
SPAN_BALANCE_COEFF = 300

# ── Solver ────────────────────────────────────────────────────────────────
OR_TOOLS_TIME_LIMIT_SEC = 20   # search time per zone

# ── Road types eligible for stop placement ────────────────────────────────
RESIDENTIAL_ROAD_TYPES = frozenset({
    "residential", "living_street",
    "secondary",   "secondary_link",
    "tertiary",    "tertiary_link",
    "unclassified",
})

# ── File paths ────────────────────────────────────────────────────────────
COMUNAS_GEOJSON  = "data/comunas_medellin.geojson"
BARRIOS_GEOJSON  = "data/barrios_medellin.geojson"
VALLE_GEOJSON    = "data/valle_aburra_urbano.geojson"
POPULATION_CSV   = "data/poblation_barrios_medellin_2026.csv"

SCENARIO_NPZ     = "outputs/scenario_data.npz"
RESULTS_JSON     = "outputs/route_results.json"
MAPS_DIR         = "outputs/maps"

# ── Visualisation ─────────────────────────────────────────────────────────
DPI = 150