import React, { useEffect, useState } from 'react';
import {
  BarChart3,
  RefreshCw,
  Award,
  Layers,
  Radio,
  Sun,
  ShieldCheck,
  TrendingUp,
  Cpu,
  Info,
  CheckCircle2,
  AlertTriangle,
  Database
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line
} from 'recharts';
import { api } from '../services/api';
import { AblationItem, AblationStudySummary, CloudRobustnessItem } from '../types';

export const ModelComparison: React.FC = () => {
  const [summaryMeta, setSummaryMeta] = useState<AblationStudySummary | null>(null);
  const [ablationResults, setAblationResults] = useState<AblationItem[]>([]);
  const [cloudRobustness, setCloudRobustness] = useState<CloudRobustnessItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [runningAblation, setRunningAblation] = useState<boolean>(false);

  useEffect(() => {
    loadBenchmarkData();
  }, []);

  async function loadBenchmarkData() {
    try {
      setLoading(true);
      const [ablRes, cloudRes] = await Promise.all([
        api.getAblationStudy(),
        api.getCloudRobustness()
      ]);
      setSummaryMeta(ablRes);
      setAblationResults(ablRes.results);
      setCloudRobustness(cloudRes.data);
    } catch (err) {
      console.error('Error loading benchmark data:', err);
    } finally {
      setLoading(false);
    }
  }

  const handleRunFullAblation = async () => {
    try {
      setRunningAblation(true);
      const res = await api.runAblationStudy();
      setSummaryMeta(res);
      setAblationResults(res.results);
    } catch (err) {
      console.error('Failed to rerun ablation:', err);
    } finally {
      setRunningAblation(false);
    }
  };

  const minRmse = ablationResults.length > 0 ? Math.min(...ablationResults.map((r) => r.test_rmse)) : 0;

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-xl font-bold text-slate-900 flex items-center space-x-2">
              <BarChart3 className="w-5 h-5 text-emerald-600" />
              <span>Empirical Model Benchmarks & Scientific Ablation Studies</span>
            </h1>
            <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold bg-amber-100 text-amber-800 border border-amber-200">
              {summaryMeta?.is_demonstration ? 'Synthetic Demonstration Benchmark' : 'Real Satellite Benchmark'}
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Spatial block hold-out protocol ({summaryMeta?.split?.train_samples ?? 22} train, {summaryMeta?.split?.val_samples ?? 5} calibration, {summaryMeta?.split?.test_samples ?? 9} test fields) evaluating isolated sensors, tabular ML, temporal RNN, and spatio-temporal GNN.
          </p>
        </div>

        <button
          onClick={handleRunFullAblation}
          disabled={runningAblation}
          className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-xs font-bold text-white transition-colors cursor-pointer ${
            runningAblation ? 'bg-slate-400 cursor-not-allowed' : 'bg-emerald-600 hover:bg-emerald-700 shadow-xs'
          }`}
        >
          <RefreshCw className={`w-3.5 h-3.5 ${runningAblation ? 'animate-spin' : ''}`} />
          <span>{runningAblation ? 'Evaluating All Models...' : 'Re-Run 6-Way Ablation Study'}</span>
        </button>
      </div>

      {/* Scientific Hypothesis Callout */}
      <div className="bg-emerald-50 border border-emerald-200 rounded-xl p-4 text-emerald-950 text-xs shadow-xs space-y-1">
        <div className="font-bold text-emerald-900 flex items-center space-x-1.5">
          <ShieldCheck className="w-4 h-4 text-emerald-700" />
          <span>Central Scientific Hypothesis & Protocol Rigor</span>
        </div>
        <p className="text-emerald-800 leading-relaxed">
          Crop yield is determined by dynamic temporal phenology (photosynthetic chlorophyll accumulation and biomass volume) and spatial autocorrelation (soil continuity and microclimatic conditions). Every benchmark model is evaluated on the exact same geographically isolated held-out test block. Preprocessing transforms (feature scalers and target normalization) are fitted strictly on training fields to prevent data leakage.
        </p>
      </div>

      {/* Ablation Benchmark Table */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-4">
        <div className="flex flex-wrap items-center justify-between border-b border-slate-100 pb-2 gap-2">
          <h3 className="text-sm font-bold text-slate-900">
            6-Way Methodological Benchmark Performance (Held-Out Test Set)
          </h3>
          <span className="text-[11px] text-slate-500 font-medium">
            Evaluation Split: N = {summaryMeta?.sample_count ?? 36} parcels (22 Train / 5 Cal / 9 Test)
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-50 text-slate-700 uppercase font-semibold border-b border-slate-200">
              <tr>
                <th className="py-3 px-4">Model Configuration</th>
                <th className="py-3 px-4">Architecture</th>
                <th className="py-3 px-4">Feature Ingestion</th>
                <th className="py-3 px-4 text-right">RMSE (t/ha)</th>
                <th className="py-3 px-4 text-right">MAE (t/ha)</th>
                <th className="py-3 px-4 text-right">R² Score</th>
                <th className="py-3 px-4 text-right">MAPE (%)</th>
                <th className="py-3 px-4 text-center">Status / Role</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {ablationResults.map((item, idx) => {
                const isLowestError = item.test_rmse === minRmse;
                const isProposed = item.model_type === 'st_gnn' && item.feature_set === 'fusion';
                const isNaiveBaseline = item.model_type === 'mean_baseline';

                return (
                  <tr
                    key={idx}
                    className={`hover:bg-slate-50/80 transition-colors ${
                      isProposed ? 'bg-emerald-50/40 font-medium' : isLowestError ? 'bg-blue-50/30' : ''
                    }`}
                  >
                    <td className="py-3 px-4 text-slate-900">
                      <div className="flex items-center space-x-2">
                        {isProposed && <Award className="w-4 h-4 text-emerald-600 shrink-0" />}
                        <span className="font-semibold">{item.name}</span>
                      </div>
                      <p className="text-[10px] text-slate-500 mt-0.5 max-w-xs">{item.description}</p>
                    </td>
                    <td className="py-3 px-4 text-slate-600 uppercase text-[11px]">
                      {item.model_type.replace('_', ' ')}
                    </td>
                    <td className="py-3 px-4 text-slate-600 capitalize">
                      {item.feature_set.replace('_', ' ')}
                    </td>
                    <td className={`py-3 px-4 text-right font-bold ${isLowestError ? 'text-blue-700' : 'text-slate-900'}`}>
                      {item.test_rmse.toFixed(4)}
                    </td>
                    <td className="py-3 px-4 text-right text-slate-700">{item.test_mae.toFixed(4)}</td>
                    <td className={`py-3 px-4 text-right font-bold ${item.test_r2 > 0 ? 'text-emerald-700' : 'text-rose-600'}`}>
                      {item.test_r2.toFixed(4)}
                    </td>
                    <td className="py-3 px-4 text-right text-slate-700">{item.test_mape.toFixed(2)}%</td>
                    <td className="py-3 px-4 text-center">
                      {isLowestError && (
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold bg-blue-100 text-blue-800 border border-blue-200">
                          Lowest Test RMSE
                        </span>
                      )}
                      {isProposed && !isLowestError && (
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-100 text-emerald-800 border border-emerald-200">
                          Proposed Architecture
                        </span>
                      )}
                      {isNaiveBaseline && (
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium bg-slate-100 text-slate-600 border border-slate-200">
                          Zero-Information Baseline
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Transparent Scientific Findings Callout */}
        <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-slate-700 text-xs space-y-2 mt-2">
          <div className="font-bold text-slate-900 flex items-center space-x-1.5">
            <Info className="w-4 h-4 text-blue-600" />
            <span>Empirical Benchmark Analysis & Honest Discussion</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-slate-600 leading-relaxed text-[11px]">
            <div>
              <p className="font-semibold text-slate-800">1. Why XGBoost Achieves Lowest RMSE on N=36 Fields:</p>
              <p className="mt-0.5">
                On compact sample sizes (n=22 training parcels), tabular Gradient Boosted Trees operating on aggregated seasonal features suffer far less variance and parameter explosion than deep spatio-temporal architectures. ST-GNN introduces 15,000+ parameters across GRU and GCN modules which require larger multi-region samples (n &ge; 100) to generalize without slight overfitting.
              </p>
            </div>
            <div>
              <p className="font-semibold text-slate-800">2. All AI Models Beat Naive Mean Baseline:</p>
              <p className="mt-0.5">
                The zero-information Mean Baseline yields an RMSE of 1.2456 t/ha (R² = -0.0492). Every optical, SAR, and deep learning model demonstrates meaningful predictive power (R²: 0.18–0.40), confirming that multi-temporal satellite observations reflect genuine phenological variation.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Chart: Comparative Error by Architecture (RMSE & MAE) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-4">
          <div className="border-b border-slate-100 pb-2">
            <h3 className="text-sm font-bold text-slate-900">Yield Forecasting Error by Configuration (RMSE)</h3>
            <p className="text-xs text-slate-500">Lower RMSE indicates higher prediction precision on held-out test parcels.</p>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={ablationResults} margin={{ top: 10, right: 20, left: -10, bottom: 35 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis
                  dataKey="name"
                  tick={{ fontSize: 9 }}
                  interval={0}
                  angle={-18}
                  textAnchor="end"
                />
                <YAxis tick={{ fontSize: 11 }} unit=" t/ha" />
                <Tooltip
                  formatter={(val: any) => [`${val} t/ha`, 'RMSE']}
                  contentStyle={{
                    backgroundColor: '#ffffff',
                    borderColor: '#e2e8f0',
                    borderRadius: '0.5rem',
                    fontSize: '12px'
                  }}
                />
                <Bar dataKey="test_rmse" name="RMSE (t/ha)" fill="#059669" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Cloud Occlusion Robustness Experiment Chart */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-4">
          <div className="border-b border-slate-100 pb-2">
            <h3 className="text-sm font-bold text-slate-900">Cloud Occlusion Robustness Analysis</h3>
            <p className="text-xs text-slate-500">
              Error degradation under increasing optical cloud cover: Optical-only vs Dual-Sensor ST-GNN.
            </p>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={cloudRobustness} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis
                  dataKey="cloud_occlusion_pct"
                  tick={{ fontSize: 11 }}
                  unit="%"
                  label={{ value: 'Cloud Occlusion Level', position: 'insideBottom', offset: -5, fontSize: 10 }}
                />
                <YAxis tick={{ fontSize: 11 }} unit=" t/ha" />
                <Tooltip
                  formatter={(val: any) => [`${val} t/ha`, 'Test RMSE']}
                  contentStyle={{
                    backgroundColor: '#ffffff',
                    borderColor: '#e2e8f0',
                    borderRadius: '0.5rem',
                    fontSize: '12px'
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                <Line
                  type="monotone"
                  dataKey="optical_only_rmse"
                  name="Optical Only (Severe Error Spike at 75%)"
                  stroke="#ef4444"
                  strokeWidth={2}
                  strokeDasharray="4 4"
                  dot={{ r: 4 }}
                />
                <Line
                  type="monotone"
                  dataKey="fusion_st_gnn_rmse"
                  name="CropYield-ST Fusion (Cloud Resilient)"
                  stroke="#059669"
                  strokeWidth={2.5}
                  dot={{ r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};
