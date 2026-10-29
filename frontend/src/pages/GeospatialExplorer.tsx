import React, { useEffect, useState } from 'react';
import {
  MapPin,
  Layers,
  Info,
  Maximize2,
  Users,
  Compass,
  CheckCircle,
  AlertCircle
} from 'lucide-react';
import { ParcelMap } from '../components/ParcelMap';
import { api } from '../services/api';
import { ParcelMetadata, ParcelDetail } from '../types';

interface GeospatialExplorerProps {
  initialParcelId?: string | null;
  onNavigateToAnalytics: (parcelId: string) => void;
}

export const GeospatialExplorer: React.FC<GeospatialExplorerProps> = ({
  initialParcelId,
  onNavigateToAnalytics
}) => {
  const [parcels, setParcels] = useState<ParcelMetadata[]>([]);
  const [selectedParcelId, setSelectedParcelId] = useState<string | null>(initialParcelId || null);
  const [parcelDetail, setParcelDetail] = useState<ParcelDetail | null>(null);
  const [colorMode, setColorMode] = useState<'yield' | 'soil' | 'default'>('yield');
  const [loadingDetail, setLoadingDetail] = useState<boolean>(false);

  useEffect(() => {
    async function fetchParcels() {
      try {
        const data = await api.getParcels();
        setParcels(data);
        if (!selectedParcelId && data.length > 0) {
          setSelectedParcelId(data[0].parcel_id);
        }
      } catch (err) {
        console.error('Error fetching parcels:', err);
      }
    }
    fetchParcels();
  }, []);

  useEffect(() => {
    if (!selectedParcelId) return;
    async function fetchDetail() {
      try {
        setLoadingDetail(true);
        const detail = await api.getParcelDetail(selectedParcelId!);
        setParcelDetail(detail);
      } catch (err) {
        console.error('Error fetching parcel details:', err);
      } finally {
        setLoadingDetail(false);
      }
    }
    fetchDetail();
  }, [selectedParcelId]);

  return (
    <div className="space-y-6">
      {/* Header and Controls */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-slate-900 flex items-center space-x-2">
            <MapPin className="w-5 h-5 text-emerald-600" />
            <span>Geospatial Agricultural Parcel Explorer</span>
          </h1>
          <p className="text-xs text-slate-500 mt-0.5">
            Interactive topological field parcel boundaries with spatial graph adjacency.
          </p>
        </div>

        {/* Layer Mode Switcher */}
        <div className="flex items-center space-x-2 bg-slate-100 p-1 rounded-lg text-xs">
          <span className="text-slate-500 font-medium px-2">Map Choropleth:</span>
          <button
            onClick={() => setColorMode('yield')}
            className={`px-3 py-1.5 rounded-md font-medium transition-colors cursor-pointer ${
              colorMode === 'yield'
                ? 'bg-white text-emerald-700 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Yield Forecast (t/ha)
          </button>
          <button
            onClick={() => setColorMode('soil')}
            className={`px-3 py-1.5 rounded-md font-medium transition-colors cursor-pointer ${
              colorMode === 'soil'
                ? 'bg-white text-amber-700 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Soil Fertility Index
          </button>
        </div>
      </div>

      {/* Main Two-Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Interactive Map */}
        <div className="lg:col-span-2 space-y-4">
          <ParcelMap
            parcels={parcels}
            selectedParcelId={selectedParcelId}
            onSelectParcel={(pid) => setSelectedParcelId(pid)}
            neighborIds={parcelDetail?.neighbors || []}
            colorMode={colorMode}
          />

          {/* Raster Imagery Notice Banner */}
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-xs text-slate-600 flex items-start space-x-2.5">
            <Info className="w-4 h-4 text-slate-500 shrink-0 mt-0.5" />
            <p>
              <strong>Scientific Data Note:</strong> Parcel polygons and time-series feature curves are fully rendered from vector boundaries and multi-sensor records. Full-scene background GeoTIFF rasters are omitted in demonstration mode to keep package size compact.
            </p>
          </div>
        </div>

        {/* Right Col: Parcel Inspector & Adjacency Relationships */}
        <div className="space-y-4">
          <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-4">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div>
                <span className="text-[10px] text-slate-400 uppercase font-semibold">Active Parcel</span>
                <h3 className="text-lg font-bold text-slate-900">{selectedParcelId || 'None'}</h3>
              </div>
              {parcelDetail?.predicted_yield && (
                <div className="bg-emerald-50 border border-emerald-200 text-emerald-800 px-3 py-1 rounded-lg text-right">
                  <span className="text-[10px] block uppercase font-medium">Predicted Yield</span>
                  <strong className="text-base font-extrabold">{parcelDetail.predicted_yield} t/ha</strong>
                </div>
              )}
            </div>

            {loadingDetail ? (
              <p className="text-xs text-slate-400 py-6 text-center">Loading parcel graph context...</p>
            ) : parcelDetail ? (
              <div className="space-y-4 text-xs">
                {/* Physical Field Metrics */}
                <div className="grid grid-cols-2 gap-2.5">
                  <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                    <span className="text-slate-400 block text-[10px] uppercase">Crop Variety</span>
                    <strong className="text-slate-900">{parcelDetail.crop_type}</strong>
                  </div>
                  <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                    <span className="text-slate-400 block text-[10px] uppercase">Parcel Extent</span>
                    <strong className="text-slate-900">{parcelDetail.area_ha} ha</strong>
                  </div>
                  <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                    <span className="text-slate-400 block text-[10px] uppercase">Ground Truth Label</span>
                    <strong className="text-slate-900">{parcelDetail.actual_yield ? `${parcelDetail.actual_yield} t/ha` : 'Unlabeled field'}</strong>
                  </div>
                  <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                    <span className="text-slate-400 block text-[10px] uppercase">Soil Quality</span>
                    <strong className="text-slate-900">{(parcelDetail.soil_quality * 100).toFixed(0)}%</strong>
                  </div>
                </div>

                {/* Satellite Observation Availability */}
                <div className="bg-slate-50 p-3 rounded-lg border border-slate-100 space-y-2">
                  <h4 className="font-semibold text-slate-900">Sensor Observation Availability</h4>
                  <div className="flex justify-between items-center text-slate-600">
                    <span>Sentinel-2 Optical (Cloud-Free):</span>
                    <span className="font-semibold text-emerald-700">{parcelDetail.optical_availability_pct}%</span>
                  </div>
                  <div className="flex justify-between items-center text-slate-600">
                    <span>Sentinel-1 SAR C-Band (All-Weather):</span>
                    <span className="font-semibold text-emerald-700">{parcelDetail.sar_availability_pct}%</span>
                  </div>
                  <div className="flex justify-between items-center text-slate-600">
                    <span>Observation Count:</span>
                    <span className="font-semibold text-slate-900">{parcelDetail.observation_count} bi-weekly steps</span>
                  </div>
                </div>

                {/* Spatial Adjacency / GCN Graph Neighbors */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <h4 className="font-semibold text-slate-900 flex items-center space-x-1.5">
                      <Users className="w-4 h-4 text-amber-600" />
                      <span>Adjacent Neighbors ({parcelDetail.neighbors.length})</span>
                    </h4>
                    <span className="text-[10px] text-slate-400">GCN Graph Edges</span>
                  </div>

                  <p className="text-[11px] text-slate-500">
                    Fields sharing contiguous topological boundaries. Highlighted on the map with dashed amber borders.
                  </p>

                  <div className="flex flex-wrap gap-1.5 pt-1">
                    {parcelDetail.neighbors.map((nbrId) => (
                      <button
                        key={nbrId}
                        onClick={() => setSelectedParcelId(nbrId)}
                        className="bg-amber-50 hover:bg-amber-100 border border-amber-200 text-amber-900 px-2 py-1 rounded text-[11px] font-medium transition-colors cursor-pointer"
                      >
                        {nbrId}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Action button */}
                <button
                  onClick={() => onNavigateToAnalytics(parcelDetail.parcel_id)}
                  className="w-full bg-emerald-600 hover:bg-emerald-700 text-white font-medium py-2.5 rounded-lg transition-colors cursor-pointer text-center"
                >
                  View Multi-Temporal Satellite Trajectories
                </button>
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
};
