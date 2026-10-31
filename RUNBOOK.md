# Reproducibility Runbook — CropYield-ST

This runbook provides unambiguous instructions to set up the environment, run each phase of the project, reproduce experimental results, and run the automated test suite.

---

## ⚡ Quick Start: One-Click Execution (Windows & macOS)

### 🪟 Windows (No Node.js / npm required)
- **Option 1 (Standalone Executable):** Double-click `CropYield-ST.exe` in the root folder.
- **Option 2 (Batch Script):** Double-click `run_demo.bat` or `run_windows.bat`.

### 🍎 macOS (No Node.js / Zero-Friction)
- **Option 1 (Native App Bundle):** Double-click `CropYield-ST.app` in macOS Finder.
- **Option 2 (Terminal Command Launcher):** Double-click `run_mac.command` in macOS Finder (or run `./run_mac.sh` in Terminal).

### 🐳 Zero-Installation Container Mode (Windows, macOS, Linux)
Requires only Docker Desktop (no Python, Node.js, C++ compilers, or GDAL installation required on your computer):
```bash
# Windows
.\run_docker.bat

# macOS (Double-click in Finder or run)
./run_docker.command
# Or standard:
docker compose up
```
The full application (React UI + PyTorch ST-GNN + FastAPI API) will automatically open at `http://localhost:8000`.

---

## ☁️ Free Cloud Hosting Deployment (Zero Local Installation)

1. **Hugging Face Spaces (Recommended for Full PyTorch AI with 16GB RAM):**
   - Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) (SDK: Docker, Blank template).
   - In terminal:
     ```bash
     git remote add hf https://huggingface.co/spaces/<YOUR_HF_USERNAME>/CropYield-ST
     git push hf main
     ```
   - Hugging Face automatically builds the Docker container and serves the full app at `https://<space-name>.hf.space`.

2. **Render.com (Free Web Service via GitHub Blueprint):**
   - In [render.com](https://render.com), click **New +** &rarr; **Blueprint** &rarr; choose `DIZIlaw/CropYield-ST`.
   - Render deploys from `render.yaml` automatically.

3. **Vercel (Frontend Global CDN):**
   - Connect repository `DIZIlaw/CropYield-ST` to [vercel.com](https://vercel.com).
   - `vercel.json` automatically builds the Vite frontend and enables client-side routing.
   - Point `VITE_API_BASE` to your hosted backend URL.

---

## 1. Prerequisites & Environment Setup

- **Python**: Version 3.10+ recommended.
- **Hardware**: Compatible with CPU execution; NVIDIA GPU with CUDA recommended for rapid deep learning training.

### Setting up Virtual Environment
```bash
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Google Earth Engine Authentication (Optional / Pre-requisite for Live Earth Engine Fetching)

To fetch live imagery from Google Earth Engine:
```bash
python -c "import ee; ee.Authenticate(); ee.Initialize()"
```
*Note: CropYield-ST includes a built-in deterministic offline data generation/loader mode in `src/gee/sentinel2.py` to allow end-to-end testing and demonstration without mandatory GEE cloud credentials.*

---

## 3. Step-by-Step Phase Execution

### Phase 1: Satellite Data Acquisition & Vegetation Indices
Run the Sentinel-2 ingestion, cloud-masking, and vegetation indices pipeline:
```bash
python scripts/run_phase1_sentinel2.py
```
Outputs produced:
- `outputs/figures/sentinel2_rgb_composite.png`
- `outputs/figures/sentinel2_ndvi_map.png`
- `outputs/figures/sentinel2_ndre_map.png`
- `outputs/figures/sentinel2_evi_map.png`
- `outputs/figures/sentinel2_ndwi_map.png`
- `outputs/figures/vegetation_indices_distributions.png`
- `data/processed/phase1_sample_indices.npz`

---

## 4. Running the Test Suite

Run all unit and integration tests:
```bash
pytest -v tests/
```
