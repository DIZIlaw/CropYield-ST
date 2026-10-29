import React, { useState, useEffect } from 'react';
import {
  Globe,
  MapPin,
  X,
  Check,
  Compass,
  Layers,
  Activity,
  ArrowRight
} from 'lucide-react';
import { api } from '../services/api';

interface LocationSwitcherModalProps {
  isOpen: boolean;
  onClose: () => void;
  onLocationChanged: () => void;
}

export const LocationSwitcherModal: React.FC<LocationSwitcherModalProps> = ({
  isOpen,
  onClose,
  onLocationChanged
}) => {
  const [presets, setPresets] = useState<any[]>([]);
  const [selectedPreset, setSelectedPreset] = useState<string>('punjab');
  const [activeTab, setActiveTab] = useState<'presets' | 'custom'>('presets');

  // Custom inputs
  const [customLat, setCustomLat] = useState<string>('29.6857');
  const [customLon, setCustomLon] = useState<string>('76.9905');
  const [customName, setCustomName] = useState<string>('Custom Farm Region');

  const [loading, setLoading] = useState<boolean>(false);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  useEffect(() => {
    async function loadPresets() {
      try {
        const data = await api.getRegions();
        setPresets(data);
      } catch (err) {
        console.error('Error fetching regional presets:', err);
      }
    }
    if (isOpen) {
      loadPresets();
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleApplyPreset = async (presetId: string) => {
    try {
      setLoading(true);
      setSuccessMsg(null);
      await api.switchRegion({ region_key: presetId });
      setSelectedPreset(presetId);
      setSuccessMsg('Location switched! Re-centering map and reloading multi-sensor data...');
      setTimeout(() => {
        setLoading(false);
        onLocationChanged();
        onClose();
      }, 900);
    } catch (err: any) {
      setLoading(false);
      alert(`Failed to switch location: ${err.message}`);
    }
  };

  const handleApplyCustom = async () => {
    const lat = parseFloat(customLat);
    const lon = parseFloat(customLon);
    if (isNaN(lat) || isNaN(lon)) {
      alert('Please enter valid numeric latitude and longitude coordinates.');
      return;
    }
    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) {
      alert('Coordinates out of range: Latitude [-90, 90], Longitude [-180, 180].');
      return;
    }

    try {
      setLoading(true);
      setSuccessMsg(null);
      await api.switchRegion({
        custom_lat: lat,
        custom_lon: lon,
        region_name: customName || `Custom Area (${lat.toFixed(2)}N, ${lon.toFixed(2)}E)`
      });
      setSuccessMsg('Custom coordinates applied! Re-centering map...');
      setTimeout(() => {
        setLoading(false);
        onLocationChanged();
        onClose();
      }, 900);
    } catch (err: any) {
      setLoading(false);
      alert(`Failed to switch location: ${err.message}`);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 backdrop-blur-xs p-4">
      <div className="bg-white rounded-2xl shadow-2xl border border-slate-200 max-w-xl w-full overflow-hidden animate-in fade-in duration-200">
        {/* Modal Header */}
        <div className="bg-slate-900 text-white px-6 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <div className="p-2 bg-emerald-600 rounded-lg">
              <Globe className="w-5 h-5 text-white" />
            </div>
            <div>
              <h3 className="font-bold text-base text-white">Change Study Area & Geographic Location</h3>
              <p className="text-xs text-slate-400">View and forecast crop yields for other agricultural regions</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white p-1 rounded-lg transition-colors cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tab Selection */}
        <div className="flex border-b border-slate-200 bg-slate-50 text-xs font-semibold">
          <button
            onClick={() => setActiveTab('presets')}
            className={`flex-1 py-3 text-center border-b-2 transition-colors cursor-pointer ${
              activeTab === 'presets'
                ? 'border-emerald-600 text-emerald-800 bg-white'
                : 'border-transparent text-slate-500 hover:text-slate-800'
            }`}
          >
            Global Agricultural Presets ({presets.length || 6})
          </button>
          <button
            onClick={() => setActiveTab('custom')}
            className={`flex-1 py-3 text-center border-b-2 transition-colors cursor-pointer ${
              activeTab === 'custom'
                ? 'border-emerald-600 text-emerald-800 bg-white'
                : 'border-transparent text-slate-500 hover:text-slate-800'
            }`}
          >
            Custom GPS Coordinates
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 max-h-[420px] overflow-y-auto space-y-4">
          {successMsg && (
            <div className="bg-emerald-50 border border-emerald-200 text-emerald-800 p-3 rounded-lg text-xs flex items-center space-x-2">
              <Check className="w-4 h-4 text-emerald-600 shrink-0" />
              <span>{successMsg}</span>
            </div>
          )}

          {activeTab === 'presets' ? (
            <div className="space-y-2.5">
              <p className="text-xs text-slate-500">
                Select an agricultural region to generate contiguous parcels and multi-sensor time-series:
              </p>
              <div className="grid grid-cols-1 gap-2.5">
                {presets.map((p) => {
                  const isSelected = selectedPreset === p.id;
                  return (
                    <div
                      key={p.id}
                      onClick={() => !loading && handleApplyPreset(p.id)}
                      className={`p-3 rounded-xl border text-xs transition-all cursor-pointer flex items-center justify-between ${
                        isSelected
                          ? 'border-emerald-500 bg-emerald-50/60 shadow-xs ring-1 ring-emerald-500'
                          : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50'
                      }`}
                    >
                      <div className="space-y-1">
                        <div className="flex items-center space-x-2">
                          <MapPin className="w-4 h-4 text-emerald-600" />
                          <span className="font-bold text-slate-900">{p.name}</span>
                        </div>
                        <p className="text-slate-500 text-[11px] pl-6">{p.description}</p>
                        <div className="pl-6 text-[10px] text-slate-400 font-mono">
                          Coordinates: {p.lat.toFixed(4)}° N, {p.lon.toFixed(4)}° E
                        </div>
                      </div>
                      <div className="shrink-0 pl-3">
                        {loading && selectedPreset === p.id ? (
                          <Activity className="w-5 h-5 text-emerald-600 animate-spin" />
                        ) : (
                          <button className="bg-emerald-600 hover:bg-emerald-700 text-white font-medium px-3 py-1.5 rounded-lg text-xs cursor-pointer flex items-center space-x-1">
                            <span>Select</span>
                            <ArrowRight className="w-3 h-3" />
                          </button>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className="space-y-4 text-xs">
              <p className="text-slate-500">
                Specify any arbitrary global latitude and longitude coordinates. The system will synthesize a contiguous agricultural field graph and multi-sensor Sentinel-1/2 time series at that location.
              </p>

              <div className="space-y-3">
                <div className="space-y-1">
                  <label className="font-semibold text-slate-700 block">Region Name / Identifier</label>
                  <input
                    type="text"
                    value={customName}
                    onChange={(e) => setCustomName(e.target.value)}
                    placeholder="e.g. Karnal Wheat Farm"
                    className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 text-slate-800 focus:ring-2 focus:ring-emerald-500"
                  />
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <label className="font-semibold text-slate-700 block">Latitude (°N/S)</label>
                    <input
                      type="number"
                      step="0.0001"
                      value={customLat}
                      onChange={(e) => setCustomLat(e.target.value)}
                      placeholder="e.g. 29.6857"
                      className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 text-slate-800 focus:ring-2 focus:ring-emerald-500 font-mono"
                    />
                  </div>

                  <div className="space-y-1">
                    <label className="font-semibold text-slate-700 block">Longitude (°E/W)</label>
                    <input
                      type="number"
                      step="0.0001"
                      value={customLon}
                      onChange={(e) => setCustomLon(e.target.value)}
                      placeholder="e.g. 76.9905"
                      className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 text-slate-800 focus:ring-2 focus:ring-emerald-500 font-mono"
                    />
                  </div>
                </div>

                <div className="bg-slate-50 p-3 rounded-lg border border-slate-200 text-slate-600 text-[11px] space-y-1">
                  <strong className="text-slate-900 block font-semibold">Examples of coordinates to try:</strong>
                  <div>• Karnal, Haryana: <code className="font-mono text-emerald-700">29.6857, 76.9905</code></div>
                  <div>• Ujjain, Madhya Pradesh: <code className="font-mono text-emerald-700">23.1765, 75.7885</code></div>
                  <div>• Iowa Corn Belt, USA: <code className="font-mono text-emerald-700">41.5868, -93.6250</code></div>
                </div>
              </div>

              <button
                onClick={handleApplyCustom}
                disabled={loading}
                className="w-full bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-2.5 rounded-lg text-xs transition-colors flex items-center justify-center space-x-2 cursor-pointer"
              >
                {loading ? (
                  <>
                    <Activity className="w-4 h-4 animate-spin" />
                    <span>Repositioning Map and Generating Area...</span>
                  </>
                ) : (
                  <>
                    <Compass className="w-4 h-4" />
                    <span>Apply Custom Location & Re-Center Map</span>
                  </>
                )}
              </button>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="bg-slate-50 px-6 py-3 border-t border-slate-200 flex justify-end">
          <button
            onClick={onClose}
            className="text-xs text-slate-600 hover:text-slate-900 font-medium px-4 py-2 rounded-lg cursor-pointer"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
