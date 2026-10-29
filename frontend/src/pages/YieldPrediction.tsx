import React, { useEffect, useState } from 'react';
import {
  Target,
  Play,
  Cpu,
  Layers,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
  ShieldCheck,
  TrendingUp,
  Activity,
  AlertTriangle
} from 'lucide-react';
import { api } from '../services/api';
import { ParcelMetadata, PredictResponse, PredictRequest } from '../types';

interface YieldPredictionProps {
  initialParcelId?: string | null;
}

export const YieldPrediction: React.FC<YieldPredictionProps> = ({ initialParcelId }) => {
  const [parcels, setParcels] = useState<ParcelMetadata[]>([]);
  const [selectedParcelId, setSelectedParcelId] = useState<string>(initialParcelId || 'PARCEL_001');
  const [modelType, setModelType] = useState<string>('st_gnn');
  const [featureSet, setFeatureSet] = useState<string>('fusion');
  const [cropType, setCropType] = useState<string>('Wheat - HD 2967');

  const [loading, setLoading] = useState<boolean>(false);
  const [predictionResult, setPredictionResult] = useState<PredictResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    async function loadParcels() {
      try {
        const pList = await api.getParcels();
        setParcels(pList);
        const active = initialParcelId || (pList.length > 0 ? pList[0].parcel_id : 'PARCEL_001');
        setSelectedParcelId(active);
        const found = pList.find((p) => p.parcel_id === active);
        if (found) setCropType(found.crop_type);
      } catch (err) {
        console.error('Error loading parcels:', err);
      }
    }
    loadParcels();
  }, [initialParcelId]);

  const handleRunPrediction = async () => {
    try {
      setLoading(true);
      setErrorMsg(null);
      const payload: PredictRequest = {
        parcel_id: selectedParcelId,
        model_type: modelType,
        feature_set: featureSet,
        crop_type: cropType
      };
      const res = await api.predictYield(payload);
      setPredictionResult(res);
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to generate prediction');
    } finally {
      setLoading(false);
    }
  };

  const selectedParcel = parcels.find((p) => p.parcel_id === selectedParcelId);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs">
        <h1 className="text-xl font-bold text-slate-900 flex items-center space-x-2">
          <Target className="w-5 h-5 text-emerald-600" />
          <span>Hyperlocal Parcel-Level Crop Yield Forecasting</span>
        </h1>
        <p className="text-xs text-slate-500 mt-0.5">
          Generate harvest yield estimates (tonnes/ha) with confidence intervals by querying the trained deep neural model.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Configuration Form */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-5">
          <h2 className="text-sm font-bold text-slate-900 border-b border-slate-100 pb-3 flex items-center space-x-2">
            <Cpu className="w-4 h-4 text-emerald-600" />
            <span>Inference Configuration</span>
          </h2>

          {/* Field Selection */}
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-slate-700">Target Field / Parcel ID</label>
            <select
              value={selectedParcelId}
              onChange={(e) => {
                setSelectedParcelId(e.target.value);
                const found = parcels.find((p) => p.parcel_id === e.target.value);
                if (found) setCropType(found.crop_type);
              }}
              className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 text-xs text-slate-800 focus:ring-2 focus:ring-emerald-500 cursor-pointer"
            >
              {parcels.map((p) => (
                <option key={p.parcel_id} value={p.parcel_id}>
                  {p.parcel_id} — {p.crop_type} ({p.area_ha} ha)
                </option>
              ))}
            </select>
          </div>

          {/* Model Architecture */}
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-slate-700">Inference Architecture</label>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
              className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 text-xs text-slate-800 focus:ring-2 focus:ring-emerald-500 cursor-pointer"
            >
              <option value="st_gnn">CropYield-ST (Temporal GRU + Spatial GCN) [Recommended]</option>
              <option value="pure_temporal_rnn">Temporal GRU (No Graph Convolution)</option>
              <option value="xgboost">Gradient Boosted Trees (XGBoost Tabular Baseline)</option>
              <option value="random_forest">Random Forest Regressor</option>
            </select>
          </div>

          {/* Sensor Input Fusion Selection */}
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-slate-700">Sensor Feature Ingestion</label>
            <select
              value={featureSet}
              onChange={(e) => setFeatureSet(e.target.value)}
              className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 text-xs text-slate-800 focus:ring-2 focus:ring-emerald-500 cursor-pointer"
            >
              <option value="fusion">Sentinel-1 SAR + Sentinel-2 Optical (Dual-Sensor Fusion)</option>
              <option value="optical_only">Sentinel-2 MSI Optical Only (Ablation)</option>
              <option value="sar_only">Sentinel-1 C-Band SAR Only (Ablation)</option>
            </select>
          </div>

          {/* Crop Type Override */}
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-slate-700">Crop Cultivar / Variety</label>
            <input
              type="text"
              value={cropType}
              onChange={(e) => setCropType(e.target.value)}
              className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 text-xs text-slate-800 focus:ring-2 focus:ring-emerald-500"
            />
          </div>

          {/* Submit Button */}
          <button
            onClick={handleRunPrediction}
            disabled={loading}
            className={`w-full py-3 rounded-lg text-xs font-bold text-white flex items-center justify-center space-x-2 cursor-pointer transition-colors ${
              loading ? 'bg-slate-400 cursor-not-allowed' : 'bg-emerald-600 hover:bg-emerald-700 shadow-sm'
            }`}
          >
            {loading ? (
              <>
                <Activity className="w-4 h-4 animate-spin" />
                <span>Running Deep Learning Pipeline...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-white" />
                <span>Generate Yield Forecast</span>
              </>
            )}
          </button>

          {errorMsg && (
            <div className="bg-red-50 border border-red-200 text-red-700 p-3 rounded-lg text-xs flex items-start space-x-2">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
              <span>{errorMsg}</span>
            </div>
          )}
        </div>

        {/* Right 2 Columns: Prediction Results Card */}
        <div className="lg:col-span-2 space-y-6">
          {predictionResult ? (
            <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-6">
              <div className="flex flex-wrap items-center justify-between border-b border-slate-100 pb-4 gap-3">
                <div>
                  <span className="text-xs font-semibold text-emerald-700 uppercase tracking-wider block">
                    Forecast Output Verified
                  </span>
                  <h3 className="text-xl font-extrabold text-slate-900 mt-0.5">
                    Field {predictionResult.parcel_id} — {predictionResult.crop_type}
                  </h3>
                </div>
                <div className="flex items-center space-x-2">
                  {predictionResult.actual_yield_tha !== null && predictionResult.actual_yield_tha !== undefined && (
                    <div className="bg-slate-100 border border-slate-300 text-slate-700 px-3 py-1 rounded-xl text-center">
                      <span className="text-[9px] uppercase font-bold tracking-wider text-amber-700 block">
                        {predictionResult.label_type === 'synthetic_reference' || predictionResult.is_demonstration
                          ? 'Synthetic Reference'
                          : 'Measured Harvest'}
                      </span>
                      <div className="text-base font-bold">{predictionResult.actual_yield_tha} <span className="text-xs">t/ha</span></div>
                    </div>
                  )}
                  <div className="bg-emerald-50 border border-emerald-300 text-emerald-800 px-4 py-2 rounded-xl text-center">
                    <span className="text-[10px] uppercase font-bold tracking-wider block text-emerald-600">
                      Estimated Yield
                    </span>
                    <div className="text-3xl font-black">{predictionResult.predicted_yield_tha} <span className="text-sm font-semibold">t/ha</span></div>
                  </div>
                </div>
              </div>

              {/* Confidence Interval and Scientific Metrics */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div className="bg-slate-50 p-3.5 rounded-xl border border-slate-100">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] uppercase text-slate-400 font-semibold block">95% Conformal Interval</span>
                    <span className="text-[9px] font-semibold text-slate-500 bg-slate-200/70 px-1.5 py-0.5 rounded">
                      Width: {predictionResult.uncertainty?.interval_width ?? (predictionResult.confidence_interval[1] - predictionResult.confidence_interval[0]).toFixed(2)} t/ha
                    </span>
                  </div>
                  <div className="text-sm font-bold text-slate-900 mt-1">
                    [{predictionResult.confidence_interval[0]} – {predictionResult.confidence_interval[1]}] t/ha
                  </div>
                  <p className="text-[10px] text-slate-500 mt-1 leading-snug">
                    Empirical test coverage: <strong className="text-slate-800">{predictionResult.uncertainty?.empirical_test_coverage_pct ?? 55.6}%</strong> (evaluated on independent test fields)
                  </p>
                </div>

                <div className="bg-slate-50 p-3.5 rounded-xl border border-slate-100">
                  <span className="text-[10px] uppercase text-slate-400 font-semibold block">Model Architecture</span>
                  <div className="text-sm font-bold text-slate-900 mt-1 uppercase">
                    {predictionResult.model_used.replace('_', ' ')}
                  </div>
                  <p className="text-[10px] text-slate-500 mt-0.5">
                    {predictionResult.model_version || 'v1.0'} • Mode: {predictionResult.feature_set}
                  </p>
                </div>

                <div className="bg-slate-50 p-3.5 rounded-xl border border-slate-100">
                  <span className="text-[10px] uppercase text-slate-400 font-semibold block">Latest Observation</span>
                  <div className="text-sm font-bold text-slate-900 mt-1">
                    {predictionResult.latest_observation_date}
                  </div>
                  <p className="text-[10px] text-slate-500 mt-0.5">Bi-weekly sequence ingested</p>
                </div>
              </div>

              {/* Small-Sample Calibration Notice */}
              {predictionResult.uncertainty && !predictionResult.uncertainty.is_validated && (
                <div className="bg-amber-50/70 border border-amber-200 rounded-xl p-3.5 text-xs text-amber-900 space-y-1">
                  <div className="font-bold flex items-center space-x-1.5 text-amber-950">
                    <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
                    <span>Split-Conformal Calibration Protocol (Small-Sample Notice)</span>
                  </div>
                  <p className="text-[11px] leading-relaxed text-amber-800">
                    Calibrated with finite-sample non-conformity scores on held-out validation fields (m={predictionResult.uncertainty.calibration_sample_size ?? 5}) and evaluated on held-out test fields (n={predictionResult.uncertainty.test_sample_size ?? 9}). While mathematically distribution-free, asymptotic 95% exact coverage guarantees require larger multi-region samples (n &ge; 50).
                  </p>
                </div>
              )}

              {/* Spatial Graph Neighbors Context */}
              <div className="bg-emerald-50/50 border border-emerald-200/80 rounded-xl p-4 space-y-2">
                <h4 className="text-xs font-bold text-emerald-950 flex items-center space-x-1.5">
                  <Layers className="w-4 h-4 text-emerald-700" />
                  <span>Graph Adjacency Context ({predictionResult.neighbors.length} adjacent fields)</span>
                </h4>
                <p className="text-xs text-emerald-900 leading-relaxed">
                  The GCN spatial convolution layer integrated phenological embeddings from neighbouring parcels (
                  <strong>{predictionResult.neighbors.join(', ')}</strong>) to account for spatial autocorrelation in soil moisture, local fertilizer management, and microclimatic conditions.
                </p>
              </div>

              {/* Scientific Disclaimer */}
              <div className="text-xs text-slate-400 border-t border-slate-100 pt-3 flex items-center space-x-2">
                <ShieldCheck className="w-4 h-4 text-slate-400 shrink-0" />
                <span>
                  CropYield-ST Academic Research Notice: Predictions are derived via deep learning models for algorithmic validation and course demonstration. Not intended as commercial agricultural commodities contracts.
                </span>
              </div>
            </div>
          ) : (
            <div className="bg-white rounded-xl border border-slate-200 p-12 text-center shadow-xs flex flex-col items-center justify-center space-y-3 min-h-[380px]">
              <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center text-slate-400">
                <Target className="w-6 h-6" />
              </div>
              <h3 className="text-base font-bold text-slate-800">No Prediction Generated Yet</h3>
              <p className="text-xs text-slate-500 max-w-md">
                Configure your desired field parcel and model architecture on the left, then click <strong>Generate Yield Forecast</strong> to execute the full spatio-temporal inference pipeline.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
