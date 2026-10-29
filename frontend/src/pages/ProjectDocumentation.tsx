import React from 'react';
import {
  BookOpen,
  Wheat,
  Cpu,
  Layers,
  Radio,
  Users,
  ShieldAlert,
  FileCheck,
  CheckCircle2,
  GitBranch
} from 'lucide-react';

export const ProjectDocumentation: React.FC = () => {
  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* Header */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-3">
        <div className="inline-flex items-center space-x-2 bg-emerald-50 text-emerald-800 text-xs px-2.5 py-1 rounded-md font-semibold border border-emerald-200">
          <BookOpen className="w-3.5 h-3.5" />
          <span>Academic Project Report & Documentation</span>
        </div>
        <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">
          CropYield-ST: Multi-Spectral Satellite & SAR Fusion for Hyperlocal Crop Yield Forecasting Using Spatio-Temporal Graph Neural Networks
        </h1>
        <div className="text-xs text-slate-500 pt-1 border-t border-slate-100 flex flex-wrap gap-4">
          <span><strong>Course:</strong> Artificial Intelligence — B.Tech CSE, Fall Semester 2026–27</span>
          <span><strong>Institution:</strong> School of Computer Science & Engineering</span>
        </div>
      </div>

      {/* Team Details */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-4">
        <h2 className="text-base font-bold text-slate-900 flex items-center space-x-2 border-b border-slate-100 pb-2">
          <Users className="w-4 h-4 text-emerald-600" />
          <span>Project Team & Responsibilities</span>
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
          <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 space-y-1">
            <strong className="text-slate-900 text-sm block">Aditya Pandey</strong>
            <span className="text-emerald-700 font-semibold block">Reg No: 24BCE2830</span>
            <p className="text-slate-500 text-[11px] mt-1">
              SAR & Optical Fusion Feature Engineering, Graph Construction, and ST-GNN PyTorch Architecture.
            </p>
          </div>
          <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 space-y-1">
            <strong className="text-slate-900 text-sm block">Rishav Raj</strong>
            <span className="text-emerald-700 font-semibold block">Reg No: 24BCE2093</span>
            <p className="text-slate-500 text-[11px] mt-1">
              Geospatial Parcel Processing, Spatial Leakage Prevention, and Baseline Regression Evaluation.
            </p>
          </div>
          <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 space-y-1">
            <strong className="text-slate-900 text-sm block">Divyansh Singh</strong>
            <span className="text-emerald-700 font-semibold block">Reg No: 24BCE2821</span>
            <p className="text-slate-500 text-[11px] mt-1">
              Full-Stack FastAPI REST API Architecture, Interactive Leaflet Frontend, and Ablation Study Engine.
            </p>
          </div>
        </div>
      </div>

      {/* 1. Problem Statement & Research Gap */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-4">
        <h2 className="text-base font-bold text-slate-900 flex items-center space-x-2 border-b border-slate-100 pb-2">
          <Wheat className="w-4 h-4 text-emerald-600" />
          <span>1. Problem Statement & Research Gap</span>
        </h2>
        <div className="text-xs text-slate-700 space-y-3 leading-relaxed">
          <p>
            Accurate, hyperlocal agricultural crop yield prediction before harvest is essential for global food security, supply chain logistics, and precision crop insurance. While remote sensing with optical satellites (e.g., Sentinel-2) provides spectral vegetation indices ($NDVI$, $NDRE$, $EVI$), it suffers from a fundamental limitation: <strong>cloud cover occlusion</strong>, which causes critical gaps during peak vegetative and flowering stages.
          </p>
          <div className="bg-amber-50 border border-amber-200 p-3 rounded-lg text-amber-900 space-y-1">
            <strong className="block font-semibold">Identified Research Gap:</strong>
            <p>
              1. Most existing agricultural yield models treat agricultural fields as independent, isolated data points (tabular ML or single-pixel CNNs), ignoring <strong>spatial autocorrelation</strong> and topological contiguity between adjacent fields sharing soil continuity, microclimate, and irrigation networks.
            </p>
            <p>
              2. Optical-only models fail catastrophically during cloudy monsoon weeks, whereas SAR observations are often utilized in isolation without multi-modal temporal fusion.
            </p>
          </div>
        </div>
      </div>

      {/* 2. System Architecture & ST-GNN Formulation */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-4">
        <h2 className="text-base font-bold text-slate-900 flex items-center space-x-2 border-b border-slate-100 pb-2">
          <Cpu className="w-4 h-4 text-emerald-600" />
          <span>2. CropYield-ST Proposed Architecture</span>
        </h2>
        <div className="text-xs text-slate-700 space-y-4 leading-relaxed">
          <p>
            CropYield-ST introduces an integrated Spatio-Temporal Graph Neural Network (ST-GNN) that unifies recurrent temporal dynamics with non-Euclidean spatial graph convolutions:
          </p>

          <div className="bg-slate-900 text-slate-200 p-4 rounded-xl font-mono text-[11px] overflow-x-auto">
            <div className="text-emerald-400 font-bold mb-2">// Logical Pipeline Flow</div>
            Sentinel-2 Optical (NDVI, NDRE, EVI, SWIR) + Sentinel-1 SAR (VV, VH, RVI, VH/VV)
            <br />
            &nbsp;&nbsp;&nbsp;&nbsp;↓
            <br />
            Temporal Recurrent Encoder (GRU/LSTM): X [N, T, F] → H_temp [N, d_temporal]
            <br />
            &nbsp;&nbsp;&nbsp;&nbsp;↓
            <br />
            Parcel Topological Adjacency Graph: A_norm = D̃^(-1/2) (A + I) D̃^(-1/2)
            <br />
            &nbsp;&nbsp;&nbsp;&nbsp;↓
            <br />
            Spatial GCN Layer: H_spatial = σ(A_norm • H_temp • W_spatial + b)
            <br />
            &nbsp;&nbsp;&nbsp;&nbsp;↓
            <br />
            Multi-Modal Feature Fusion + Skip Connection: H_fused = [H_temp || H_spatial]
            <br />
            &nbsp;&nbsp;&nbsp;&nbsp;↓
            <br />
            Regression Head (MLP with LayerNorm & Dropout) → Predicted Yield (tonnes/ha)
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100 space-y-1">
              <strong className="text-slate-900 block font-semibold">Temporal Component (GRU)</strong>
              <p className="text-slate-500 text-[11px]">
                Captures dynamic phenological trajectories across successive observation dates, modeling rapid biomass accumulation during heading and chlorophyll degradation during senescence.
              </p>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100 space-y-1">
              <strong className="text-slate-900 block font-semibold">Spatial Component (GCN)</strong>
              <p className="text-slate-500 text-[11px]">
                Propagates contextual information across contiguous field boundaries using symmetric normalized graph Laplacian, modeling regional soil quality and local agricultural management.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* 3. Mathematical Formulations */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-4">
        <h2 className="text-base font-bold text-slate-900 flex items-center space-x-2 border-b border-slate-100 pb-2">
          <Layers className="w-4 h-4 text-emerald-600" />
          <span>3. Mathematical Index Formulations</span>
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div className="p-3 bg-slate-50 rounded-lg border border-slate-100 space-y-1.5">
            <span className="font-bold text-slate-800">Normalized Difference Vegetation Index (NDVI)</span>
            <div className="font-mono bg-white p-2 rounded border border-slate-200 text-emerald-700">
              NDVI = (NIR - Red) / (NIR + Red)
            </div>
            <p className="text-slate-500 text-[11px]">Measures chlorophyll absorption and photosynthetic capacity.</p>
          </div>

          <div className="p-3 bg-slate-50 rounded-lg border border-slate-100 space-y-1.5">
            <span className="font-bold text-slate-800">Normalized Difference Red Edge (NDRE)</span>
            <div className="font-mono bg-white p-2 rounded border border-slate-200 text-blue-700">
              NDRE = (NIR - RedEdge) / (NIR + RedEdge)
            </div>
            <p className="text-slate-500 text-[11px]">Avoids saturation in dense, mid-to-late season crop canopies.</p>
          </div>

          <div className="p-3 bg-slate-50 rounded-lg border border-slate-100 space-y-1.5">
            <span className="font-bold text-slate-800">SAR Decibel Conversion</span>
            <div className="font-mono bg-white p-2 rounded border border-slate-200 text-slate-800">
              σ⁰_dB = 10 • log₁₀(max(σ⁰_linear, ε))
            </div>
            <p className="text-slate-500 text-[11px]">Compresses dynamic radar backscatter response range.</p>
          </div>

          <div className="p-3 bg-slate-50 rounded-lg border border-slate-100 space-y-1.5">
            <span className="font-bold text-slate-800">Dual-Polarization Radar Vegetation Index (RVI)</span>
            <div className="font-mono bg-white p-2 rounded border border-slate-200 text-purple-700">
              RVI = (4 • VH) / (VV + VH)
            </div>
            <p className="text-slate-500 text-[11px]">Directly correlates with canopy volume scattering and fresh biomass.</p>
          </div>
        </div>
      </div>

      {/* 4. Academic References */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-4">
        <h2 className="text-base font-bold text-slate-900 flex items-center space-x-2 border-b border-slate-100 pb-2">
          <BookOpen className="w-4 h-4 text-emerald-600" />
          <span>4. Academic References</span>
        </h2>
        <ol className="list-decimal list-inside text-xs text-slate-600 space-y-2 leading-relaxed">
          <li>
            Kipf, T. N., & Welling, M. (2017). <em>Semi-Supervised Classification with Graph Convolutional Networks</em>. International Conference on Learning Representations (ICLR).
          </li>
          <li>
            You, J., Li, X., Low, M., Lobell, D., & Ermon, S. (2017). <em>Deep Gaussian Process for Crop Yield Prediction Based on Remote Sensing Data</em>. AAAI Conference on Artificial Intelligence.
          </li>
          <li>
            Khaki, S., & Wang, L. (2019). <em>Crop Yield Prediction Using Deep Neural Networks</em>. Frontiers in Plant Science, 10, 621.
          </li>
          <li>
            Mandal, D., Hosseini, M., McNairn, H., Kumar, V., & Rao, Y. S. (2020). <em>An Assessment of Dual-Polarimetric SAR Vegetation Indices for Crop Phenology Monitoring</em>. Remote Sensing of Environment, 247, 111903.
          </li>
          <li>
            Gao, B. C. (1996). <em>NDWI—A Normalized Difference Water Index for Remote Sensing of Vegetation Liquid Water From Space</em>. Remote Sensing of Environment, 58(3), 257-266.
          </li>
        </ol>
      </div>
    </div>
  );
};
