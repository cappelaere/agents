Here are **10 Arctic-specific operational watch queries** designed for the **NSIDC Sea Ice Agent API** described in `seaice_openapi.yaml` .
These are phrased as **operator intent**, not code — ready for assistant task orchestration or COP dashboards.

---

### ✅ **10 User Queries (Arctic Region)**

1. **“What is the current sea-ice concentration near Utqiaġvik (Barrow) today?”**
   *(Uses `/point` with lat/lon for Utqiaġvik.)*

2. **“Show me sea-ice conditions across the Chukchi Sea for the last 24 hours.”**
   *(Use `/wms` to fetch visualization + `/download` for current dataset.)*

3. **“Compute the average ice concentration north of 70° latitude on today’s date.”**
   *(Use `/stats` with bbox `[70, -180, 90, 180]`.)*

4. **“What is the sea-ice concentration along the Northern Sea Route near Pevek for this week?”**
   *(Repeated `/point` queries or `/stats` over corridor.)*

5. **“Show a WMS tile layer I can load into my map client to view Arctic ice conditions.”**
   *(Call `/wms` to retrieve the `wms_url_template`.)*

6. **“Preload the latest sea-ice dataset so it’s cached before satellite downlink delay hits.”**
   *(Use `/download` with no time parameter → defaults to most recent available.)*

7. **“What was the ice concentration at the Bering Strait choke point on 2025-09-01?”**
   *(Call `/point` with specified date + lat/lon around 65.7° N / -168.9° W.)*

8. **“Produce mean, min, and max ice concentration for the Beaufort Sea box today.”**
   *(Use `/stats` with bbox like `[68, -165, 75, -135]`.)*

9. **“Which sensor was used to generate yesterday’s dataset and was it cached or downloaded?”**
   *(Call `/download?time=YYYY-MM-DD` to inspect `sensor` and `status` fields.)*

10. **“Is the sea-ice agent online and which satellite sources are currently being used?”**
    *(Call `/health` to view `sensors_tried` and `governance` metadata.)*

---

### 🌍 **Example Region Bounding Boxes for `/stats`**

| Region                 | BBox                   | Notes                                     |
| ---------------------- | ---------------------- | ----------------------------------------- |
| High Arctic Basin      | `[80, -180, 90, 180]`  | Pack ice core                             |
| Chukchi Sea            | `[67, -180, 75, -150]` | Operational shipping + fisheries interest |
| Beaufort Sea           | `[68, -165, 75, -135]` | U.S. Arctic offshore                      |
| Bering Strait Corridor | `[64, -170, 67, -160]` | Strategic choke point                     |

---

### 🧭 Want these turned into **real API calls** next?

Choose your output format:

| Option                                                     | Output Style |
| ---------------------------------------------------------- | ------------ |
| **A)** `curl` commands for watch floor operators           |              |
| **B)** Python (`requests` or `httpx`) for agents           |              |
| **C)** WatsonX Orchestrate **tool-call JSON**              |              |
| **D)** Map-ready tile URLs for **Leaflet / Cesium / QGIS** |              |
| **E)** All of the above                                    |              |

**Reply with A, B, C, D, or E.**
