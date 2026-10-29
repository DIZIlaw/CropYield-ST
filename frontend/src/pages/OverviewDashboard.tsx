import React, { useEffect, useState } from 'react';
import {
  Wheat,
  MapPin,
  Calendar,
  Layers,
  Award,
  AlertTriangle,
  ArrowRight,
  TrendingUp,
  Activity,
  CheckCircle2
} from 'lucide-react';
import { StatCard } from '../components/StatCard';
import { ParcelMap } from '../components/ParcelMap';
import { api } from '../services/api';
import { DashboardSummary, ParcelMetadata, ExperimentRecord } from '../types';

interface OverviewDashboardProps {
  onNavigate: (tab: string, parcelId?: string) => void;
}

export const OverviewDashboard: React.FC<OverviewDashboardProps> = ({ onNavigate }) => {
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [parcels, setParcels] = useState<ParcelMetadata[]>([]);
  const [experiments, setExperiments] = useState<ExperimentRecord[]>([]);
  const [selectedParcelId, setSelectedParcelId] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        const [sumRes, parRes, expRes] = await Promise.all([
          api.getSummary(),
          api.getParcels(),
          api.getExperiments()
        ]);
        setSummary(sumRes);
        setParcels(parRes);
        setExperiments(expRes.slice(0, 5));
        if (parRes.length > 0) setSelectedParcelId(parRes[0].parcel_id);
      } catch (err) {
        console.error('Error loading dashboard data:', err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center space-y-3">
          <Activity className="w-8 h-8 text-emerald-600 animate-spin mx-auto" />
          <p className="text-slate-600 font-medium text-sm">Loading CropYield-ST AI Dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Hero Header */}
      <div className="bg-gradient-to-r from-slate-900 to-slate-800 text-white rounded-2xl p-6 sm:p-8 shadow-sm border border-slate-700/60">
        <div className="max-w-4xl space-y-3">
          <div className="inline-flex items-center space-x-2 bg-emerald-500/20 border border-emerald-500/30 text-emerald-300 text-xs px-3 py-1 rounded-full font-medium">
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>Academic AI Project — Fall 2026–27</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white">
            Multi-Spectral Satellite & SAR Fusion for Hyperlocal Crop Yield Forecasting
          </h1>
          <p className="text-slate-300 text-sm leading-relaxed">
            CropYield-ST integrates multi-temporal Sentinel-2 optical surface reflectance, Sentinel-1 Synthetic Aperture Radar (SAR) backscatter, temporal GRU encoders, and parcel-adjacency Graph Convolutional Networks (GCN) to forecast agricultural yields at individual field resolution.
          </p>
          <div className="pt-2 flex flex-wrap items-center gap-4 text-xs text-slate-400">
            <div>
              <span className="text-slate-500">Team: </span>
              <strong className="text-slate-200">Aditya Pandey</strong> (24BCE2830) • <strong className="text-slate-200">Rishav Raj</strong> (24BCE2093) • <strong className="text-slate-200">Divyansh Singh</strong> (24BCE2821)
            </div>
            <div className="border-l border-slate-700 pl-4">
              <span className="text-slate-500">Course: </span>
              <span className="text-slate-300">Artificial Intelligence — B.Tech CSE</span>
            </div>
          </div>
        </div>
      </div>

      {/* Demonstration notice alert */}
      {summary?.is_demonstration && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 flex items-start space-x-3 text-amber-900 shadow-xs">
          <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
          <div className="text-xs space-y-1">
            <div className="font-semibold text-amber-950">Demonstration Mode & Provenance Notice</div>
            <p className="text-amber-800 leading-relaxed">
              This deployment is running on calibrated multi-sensor synthetic observation sequences simulating a 180-day wheat growing cycle in Punjab, India. This guarantees 100% reproducible execution of the deep learning pipeline, parcel graph construction, and ablation benchmarks without requiring live cloud credentials.
            </p>
          </div>
        </div>
      )}

      {/* Primary KPI Metrics Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Monitored Parcels"
          value={summary?.total_parcels ?? 0}
          subtitle="Contiguous agricultural fields"
          icon={Layers}
          color="emerald"
        />
        <StatCard
          title="Satellite Observations"
          value={summary?.total_observations ?? 0}
          subtitle="Sentinel-1 & Sentinel-2 aligned"
          icon={Calendar}
          color="blue"
        />
        <StatCard
          title="Mean Forecast Yield"
          value={summary?.mean_predicted_yield ? `${summary.mean_predicted_yield} t/ha` : 'N/A'}
          subtitle={summary?.mean_actual_yield ? `Ground-truth: ${summary.mean_actual_yield} t/ha` : 'Awaiting training'}
          icon={TrendingUp}
          color="indigo"
        />
        <StatCard
          title="Trained ST-GNN Models"
          value={summary?.trained_models_count ?? 1}
          subtitle="Validated without spatial leakage"
          icon={Award}
          color="purple"
        />
      </div>

      {/* Main Grid: Interactive Map Preview + Quick Details */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Geospatial Parcel Map Preview */}
        <div className="lg:col-span-2 bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-3">
            <div>
              <h2 className="text-base font-bold text-slate-900 flex items-center space-x-2">
                <MapPin className="w-4 h-4 text-emerald-600" />
                <span>Geospatial Field Parcel Study Area</span>
              </h2>
              <p className="text-xs text-slate-500">
                Interactive parcel boundaries. Click any field to inspect yield predictions and adjacent neighbours.
              </p>
            </div>
            <button
              onClick={() => onNavigate('explorer', selectedParcelId || undefined)}
              className="text-xs text-emerald-700 font-semibold hover:text-emerald-800 flex items-center space-x-1 cursor-pointer"
            >
              <span>Open Full Explorer</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <ParcelMap
            parcels={parcels}
            selectedParcelId={selectedParcelId}
            onSelectParcel={(pid) => setSelectedParcelId(pid)}
            colorMode="yield"
          />
        </div>

        {/* Right Col: Selected Parcel Quick Stats & Benchmarks */}
        <div className="space-y-6">
          {/* Selected Field Quick Card */}
          <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-4">
            <h3 className="text-sm font-bold text-slate-900 border-b border-slate-100 pb-2">
              Selected Parcel Overview
            </h3>

            {selectedParcelId ? (
              (() => {
                const p = parcels.find((item) => item.parcel_id === selectedParcelId);
                if (!p) return <p className="text-xs text-slate-400">Select a parcel on the map.</p>;
                return (
                  <div className="space-y-3 text-xs">
                    <div className="flex justify-between items-center bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                      <span className="font-semibold text-slate-700">{p.parcel_id}</span>
                      <span className="bg-emerald-100 text-emerald-800 font-bold px-2 py-0.5 rounded text-[11px]">
                        {p.predicted_yield ? `${p.predicted_yield} t/ha` : 'Forecast Pending'}
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-2 text-slate-600">
                      <div className="bg-slate-50/80 p-2 rounded border border-slate-100">
                        <span className="text-slate-400 block text-[10px] uppercase">Crop Variety</span>
                        <strong className="text-slate-900">{p.crop_type}</strong>
                      </div>
                      <div className="bg-slate-50/80 p-2 rounded border border-slate-100">
                        <span className="text-slate-400 block text-[10px] uppercase">Parcel Area</span>
                        <strong className="text-slate-900">{p.area_ha} hectares</strong>
                      </div>
                      <div className="bg-slate-50/80 p-2 rounded border border-slate-100">
                        <span className="text-slate-400 block text-[10px] uppercase">Soil Quality</span>
                        <strong className="text-slate-900">{(p.soil_quality * 100).toFixed(0)}% fertility</strong>
                      </div>
                      <div className="bg-slate-50/80 p-2 rounded border border-slate-100">
                        <span className="text-slate-400 block text-[10px] uppercase">Graph Degree</span>
                        <strong className="text-slate-900">{p.neighbors_count} neighbours</strong>
                      </div>
                    </div>

                    <div className="pt-2 flex flex-col space-y-2">
                      <button
                        onClick={() => onNavigate('analytics', p.parcel_id)}
                        className="w-full bg-slate-900 hover:bg-slate-800 text-white font-medium py-2 rounded-lg text-xs transition-colors cursor-pointer"
                      >
                        View Optical & SAR Trajectories
                      </button>
                      <button
                        onClick={() => onNavigate('prediction', p.parcel_id)}
                        className="w-full bg-emerald-600 hover:bg-emerald-700 text-white font-medium py-2 rounded-lg text-xs transition-colors cursor-pointer"
                      >
                        Simulate Yield Forecast
                      </button>
                    </div>
                  </div>
                );
              })()
            ) : (
              <p className="text-xs text-slate-400">Click a parcel polygon on the map.</p>
            )}
          </div>

          {/* Model Experiment Summary Card */}
          <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-3">
            <div className="flex items-center justify-between border-b border-slate-100 pb-2">
              <h3 className="text-sm font-bold text-slate-900">Recent Experiments</h3>
              <button
                onClick={() => onNavigate('ablation')}
                className="text-[11px] text-emerald-700 hover:underline font-semibold cursor-pointer"
              >
                View Ablation Study
              </button>
            </div>

            {experiments.length > 0 ? (
              <div className="space-y-2">
                {experiments.map((exp) => (
                  <div
                    key={exp.experiment_id}
                    className="p-2.5 rounded-lg border border-slate-100 bg-slate-50/50 flex items-center justify-between text-xs"
                  >
                    <div>
                      <div className="font-semibold text-slate-900 uppercase tracking-tight">
                        {exp.model_type.replace('_', ' ')}
                      </div>
                      <div className="text-[10px] text-slate-500">
                        Features: {exp.feature_set} • {exp.duration_sec}s
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="font-bold text-emerald-700">
                        RMSE: {exp.test_rmse ? `${exp.test_rmse} t/ha` : 'N/A'}
                      </div>
                      <div className="text-[10px] text-slate-500">
                        R²: {exp.test_r2 ?? 'N/A'}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-slate-400 py-3 text-center">No training experiments recorded yet.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
