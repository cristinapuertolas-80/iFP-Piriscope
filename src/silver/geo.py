"""Utilidades geográficas comunes del cruce espacial de Silver."""

from __future__ import annotations

import numpy as np

RADIO_TIERRA_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Distancia ortodrómica en km (admite arrays con *broadcasting*)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * RADIO_TIERRA_KM * np.arcsin(np.sqrt(a))


def mas_cercano(lat: np.ndarray, lon: np.ndarray, lat_ref: np.ndarray,
                lon_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Para cada punto (lat, lon), índice del punto de referencia más cercano y su distancia."""
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    d = haversine_km(lat[:, None], lon[:, None], np.asarray(lat_ref)[None, :], np.asarray(lon_ref)[None, :])
    idx = d.argmin(axis=1)
    return idx, d[np.arange(len(lat)), idx]
