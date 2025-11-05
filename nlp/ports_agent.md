Here are **10 Alaska-focused natural language queries** designed for the **Ports Agent** (UN/LOCODE lookup), each of which maps directly to **`GET /search`** from the OpenAPI spec. 

These work *as-is* with **WatsonX Orchestrate** or your **ports_agent** endpoint.

---

### ✅ **10 Natural Queries (Arctic / Alaska Focus)**

| #  | User Query (Natural Language)                                   | Corresponding API Call                                         |
| -- | --------------------------------------------------------------- | -------------------------------------------------------------- |
| 1  | “Find the UN/LOCODE for Nome, AK.”                               | `/search?name=Nome&state=AK&country=US&limit=5`                |
| 2  | “Lookup Barrow in Alaska.”                                       | `/search?name=Barrow&state=AK&country=US&limit=5`              |
| 3  | “What is the port code for Dutch Harbor?”                        | `/search?name=Dutch%20Harbor&state=AK&country=US&limit=5`      |
| 4  | “Search for Kotzebue port code.”                                 | `/search?name=Kotzebue&state=AK&country=US&limit=5`            |
| 5  | “Give me ports in Alaska that contain ‘Bay’ in their name.”      | `/search?name=Bay&state=AK&country=US&limit=20&min_score=70`   |
| 6  | “Find the UN/LOCODE for Kodiak.”                                 | `/search?name=Kodiak&state=AK&country=US&limit=5`              |
| 7  | “Look up port information for Adak (Aleutians).”                 | `/search?name=Adak&state=AK&country=US&limit=5`                |
| 8  | “Return ports in Southeast Alaska near Juneau.”                  | `/search?name=Juneau&state=AK&country=US&limit=5`              |
| 9  | “What is the UN/LOCODE for Anchorage harbor?”                    | `/search?name=Anchorage&state=AK&country=US&limit=5`           |
| 10 | “Find ports containing ‘Sound’ in Alaska.”                       | `/search?name=Sound&state=AK&country=US&limit=20&min_score=70` |


