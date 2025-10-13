## 🧩 Step 1 – Create the Virtual Environment

From your project root (where your agent code lives):

```bash
python3 -m venv venv
```

This creates a new isolated environment in a folder named `venv/`.
You can name it anything (e.g. `.env`, `.venv`, `agent_env`), but `venv` is standard.

---

## 🧠 Step 2 – Activate the Environment

### On **macOS / Linux**

```bash
source venv/bin/activate
```

### On **Windows (PowerShell)**

```powershell
venv\Scripts\Activate.ps1
```

Once activated, your shell prompt will change — e.g.

```
(venv) user@host:~/project$
```

All packages you install now go *inside* this environment, not system-wide.

---

## ⚙️ Step 3 – Upgrade pip (optional but recommended)

```bash
pip install --upgrade pip
```

---

## 📦 Step 4 – Install Required Libraries for Your Agents

For the **Open-Meteo + NSIDC + Governance PoX** agents, you’ll likely need:

```bash
pip install fastapi uvicorn httpx pydantic
pip install xarray netCDF4 pydap pyproj rioxarray numpy
pip install python-dotenv
```

If you’ll add LLM or Watsonx integration later:

```bash
pip install openai ibm-watsonx-ai
```

---

## 🧾 Step 5 – Freeze Requirements (for portability)

After verifying everything runs:

```bash
pip freeze > requirements.txt
```

This captures all exact library versions — useful for deployments or container builds.

---

## 🚀 Step 6 – Run Your Agentic API Locally

Example:

```bash
uvicorn metoc_openmeteo_agent:app --port 8080 --reload
```

Then test:

```bash
curl http://localhost:8080/metoc/healthz
```

---

## 🧹 Step 7 – Deactivate When Done

```bash
deactivate
```

# macOS/Linux
Store Earhtdata login in .netrc
printf "machine urs.earthdata.nasa.gov login $USERNAME password $PASSWORD$\n" >> ~/.netrc
chmod 600 ~/.netrc
