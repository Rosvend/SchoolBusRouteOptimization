"""Routing zones used by the CVRPTW adapter (ported from naomi:CVRPTW_optimización/config.py)."""

ZONE_DEFINITIONS = [
    {"zone_name": "Norte",         "color": "#3A86FF",
     "members": ["Bello"]},
    {"zone_name": "Nororiental",   "color": "#FF006E",
     "members": ["Comuna 1 - Popular", "Comuna 2 - Santa Cruz", "Comuna 3 - Manrique"]},
    {"zone_name": "Noroccidental", "color": "#FB5607",
     "members": ["Comuna 4 - Aranjuez", "Comuna 5 - Castilla",
                 "Comuna 6 - Doce de Octubre", "Comuna 7 - Robledo"]},
    {"zone_name": "Centro",        "color": "#FFBE0B",
     "members": ["Comuna 10 - La Candelaria", "Comuna 11 - Laureles-Estadio",
                 "Comuna 12 - La América",   "Comuna 13 - San Javier"]},
    {"zone_name": "Suroriental",   "color": "#8338EC",
     "members": ["Comuna 8 - Villa Hermosa", "Comuna 9 - Buenos Aires",
                 "Comuna 14 - El Poblado",   "Envigado"]},
    {"zone_name": "Sur",           "color": "#06D6A0",
     "members": ["Comuna 15 - Guayabal", "Comuna 16 - Belén",
                 "Itagüí", "Sabaneta", "La Estrella"]},
]

# CVRPTW parameters (from naomi's config)
MAX_ROUTE_MINUTES   = 75
BOARDING_SECONDS    = 90
SPAN_BALANCE_COEFF  = 300
SPEED_FALLBACK_KPH  = 30
OR_TOOLS_TIME_LIMIT_SEC = 20
