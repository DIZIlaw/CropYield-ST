import React, { useEffect, useState } from 'react';
import {
  LineChart as LineChartIcon,
  Cloud,
  Sun,
  Radio,
  Layers,
  Info,
  Calendar,
  AlertCircle
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar
} from 'recharts';
import { api } from '../services/api';
import { ParcelMetadata, TimeSeriesObservation } from '../types';

interface SatelliteAnalyticsProps {
  initialParcelId?: string | null;
}

export const SatelliteAnalytics: React.FC<SatelliteAnalyticsProps> = ({ initialParcelId }) => {
  const [parcels, setParcels] = useState<ParcelMetadata[]>([]);
  const [selectedParcelId, setSelectedParcelId] = useState<string>(initialParcelId || 'PARCEL_001');
  const [observations, setObservations] = useState<TimeSeriesObservation[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    async function init() {
      try {
        const pList = await api.getParcels();
        setParcels(pList);
        const activeId = initialParcelId || (pList.length > 0 ? pList[0].parcel_id : 'PARCEL_001');
        setSelectedParcelId(activeId);
      } catch (err) {
        console.error('Error initializing analytics:', err);
      }
    }
    init();
  }, [initialParcelId]);

  useEffect(() => {
    if (!selectedParcelId) return;
    async function loadTimeSeries() {
      try {
        setLoading(true);
        const res = await api.getParcelTimeSeries(selectedParcelId);
        setObservations(res.observations);
      } catch (err) {
        console.error('Error fetching timeseries:', err);
      } finally {
        setLoading(false);
      }
    }
    loadTimeSeries();
  }, [selectedParcelId]);

  // Optical cloud statistics
  const totalObs = observations.length;
  const cloudyObs = observations.filter((o) => !o.optical_valid || o.cloud_masked).length;
  const validOpticalPct = totalObs > 0 ? Math.round(((totalObs - cloudyObs) / totalObs) * 100) : 100;

  return (
    <div className="space-y-6">
      {/* Top Header & Field Selector */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-slate-900 flex items-center space-x-2">
            <LineChartIcon className="w-5 h-5 text-emerald-600" />
            <span>Multi-Sensor Satellite Analytics & Phenology Trajectories</span>
          </h1>
          <p className="text-xs text-slate-500 mt-0.5">
            Synchronized Sentinel-2 MSI Optical Spectral Indices & Sentinel-1 C-Band SAR Dual-Polarization Dynamics.
          </p>
        </div>

        {/* Parcel Selector */}
        <div className="flex items-center space-x-2">
          <label className="text-xs font-semibold text-slate-600">Select Field:</label>
          <select
            value={selectedParcelId}
            onChange={(e) => setSelectedParcelId(e.target.value)}
            className="bg-slate-50 border border-slate-300 rounded-lg px-3 py-1.5 text-xs font-medium text-slate-800 focus:outline-hidden focus:ring-2 focus:ring-emerald-500 cursor-pointer"
          >
            {parcels.map((p) => (
              <option key={p.parcel_id} value={p.parcel_id}>
                {p.parcel_id} ({p.crop_type})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Cloud Robustness Explanatory Banner */}
      <div className="bg-blue-50/70 border border-blue-200 rounded-xl p-4 flex items-start space-x-3 text-blue-900 text-xs shadow-xs">
        <Radio className="w-5 h-5 text-blue-600 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <div className="font-semibold text-blue-950">
            Physical Insight: Why SAR Microwave Backscatter Complements Optical Indices
          </div>
          <p className="text-blue-800 leading-relaxed">
            Optical sensors (Sentinel-2) measure surface reflectance in the visible and near-infrared bands (400–850 nm), which scatter strongly in cloud water droplets. Sentinel-1 SAR operates in the microwave C-band (λ ≈ 5.6 cm), penetrating cloud layers and monsoon storms unimpeded. While SAR does not directly measure chlorophyll, its cross-polarization ($VH$) directly quantifies volume scattering from canopy structure and stalks.
          </p>
        </div>
      </div>

      {/* Observation Availability Summary KPI */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-xs">
          <span className="text-[10px] uppercase font-bold text-slate-400 block">Optical Availability</span>
          <div className="text-2xl font-extrabold text-emerald-700 mt-1">{validOpticalPct}%</div>
          <p className="text-[11px] text-slate-500 mt-0.5">
            {totalObs - cloudyObs} of {totalObs} dates cloud-free
          </p>
        </div>

        <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-xs">
          <span className="text-[10px] uppercase font-bold text-slate-400 block">SAR Radar Availability</span>
          <div className="text-2xl font-extrabold text-blue-700 mt-1">100%</div>
          <p className="text-[11px] text-slate-500 mt-0.5">
            All-weather acquisition throughout growing season
          </p>
        </div>

        <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-xs">
          <span className="text-[10px] uppercase font-bold text-slate-400 block">Occluded Periods</span>
          <div className="text-2xl font-extrabold text-amber-600 mt-1">{cloudyObs} dates</div>
          <p className="text-[11px] text-slate-500 mt-0.5">
            Cloud cover &gt; 50% successfully bridged by SAR
          </p>
        </div>
      </div>

      {/* Chart 1: Optical Vegetation Indices (NDVI, NDRE, EVI, NDWI) */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-3">
          <div>
            <h3 className="text-sm font-bold text-slate-900 flex items-center space-x-2">
              <Sun className="w-4 h-4 text-emerald-600" />
              <span>Optical Spectral Indices Trajectory (Sentinel-2 MSI)</span>
            </h3>
            <p className="text-xs text-slate-500">
              Normalized Difference Vegetation Index (NDVI), Red Edge (NDRE), and Enhanced Vegetation Index (EVI).
            </p>
          </div>
          <div className="text-[11px] text-slate-500 bg-slate-50 px-2.5 py-1 rounded border border-slate-200">
            Range: [-0.2, 1.0]
          </div>
        </div>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={observations} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis domain={[-0.1, 1.0]} tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#ffffff',
                  borderColor: '#e2e8f0',
                  borderRadius: '0.5rem',
                  fontSize: '12px'
                }}
              />
              <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '8px' }} />
              <Line
                type="monotone"
                dataKey="ndvi"
                name="NDVI (Photosynthetic Active Biomass)"
                stroke="#059669"
                strokeWidth={2.5}
                dot={{ r: 3 }}
                activeDot={{ r: 6 }}
              />
              <Line
                type="monotone"
                dataKey="ndre"
                name="NDRE (Red Edge Chlorophyll Density)"
                stroke="#2563eb"
                strokeWidth={2}
                dot={{ r: 3 }}
              />
              <Line
                type="monotone"
                dataKey="evi"
                name="EVI (Atmosphere-Corrected Index)"
                stroke="#d97706"
                strokeWidth={1.5}
                strokeDasharray="4 4"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Chart 2: Sentinel-1 SAR Radar Backscatter (VV, VH in dB and Ratios) */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-3">
          <div>
            <h3 className="text-sm font-bold text-slate-900 flex items-center space-x-2">
              <Radio className="w-4 h-4 text-blue-600" />
              <span>Sentinel-1 SAR C-Band Backscatter Intensity (dB & Ratio)</span>
            </h3>
            <p className="text-xs text-slate-500">
              Co-polarization ($VV$), Cross-polarization ($VH$), and Radar Vegetation Index ($RVI$).
            </p>
          </div>
          <div className="text-[11px] text-slate-500 bg-slate-50 px-2.5 py-1 rounded border border-slate-200">
            Units: Decibels (dB)
          </div>
        </div>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={observations} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis domain={[-25, -5]} tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#ffffff',
                  borderColor: '#e2e8f0',
                  borderRadius: '0.5rem',
                  fontSize: '12px'
                }}
              />
              <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '8px' }} />
              <Line
                type="monotone"
                dataKey="sar_vv_db"
                name="VV Backscatter (dB) - Surface & Stalks"
                stroke="#475569"
                strokeWidth={2}
                dot={{ r: 3 }}
              />
              <Line
                type="monotone"
                dataKey="sar_vh_db"
                name="VH Backscatter (dB) - Canopy Volume"
                stroke="#2563eb"
                strokeWidth={2.5}
                dot={{ r: 3 }}
                activeDot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Chart 3: Cloud Cover and Masking Metadata Bar Chart */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-4">
        <div className="border-b border-slate-100 pb-3">
          <h3 className="text-sm font-bold text-slate-900 flex items-center space-x-2">
            <Cloud className="w-4 h-4 text-slate-600" />
            <span>Cloud Occlusion and Atmospheric Masking Timeline</span>
          </h3>
          <p className="text-xs text-slate-500">
            Shows dates where cloud obscuration flags triggered SAR interpolation bridging.
          </p>
        </div>

        <div className="h-44 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={observations} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} unit="%" />
              <Tooltip
                formatter={(val: any) => [`${val}%`, 'Cloud Cover']}
                contentStyle={{
                  backgroundColor: '#ffffff',
                  borderColor: '#e2e8f0',
                  borderRadius: '0.5rem',
                  fontSize: '12px'
                }}
              />
              <Bar dataKey="cloud_cover_percent" name="Scene Cloud Cover (%)" fill="#94a3b8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
