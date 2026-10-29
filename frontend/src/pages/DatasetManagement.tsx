import React, { useEffect, useState } from 'react';
import {
  Database,
  UploadCloud,
  FileText,
  CheckCircle,
  AlertTriangle,
  Info,
  Layers,
  Download,
  Activity,
  Globe,
  ShieldCheck
} from 'lucide-react';
import { api } from '../services/api';

export const DatasetManagement: React.FC = () => {
  const [schemaInfo, setSchemaInfo] = useState<any>(null);
  const [activeDataset, setActiveDataset] = useState<any>(null);
  const [providersStatus, setProvidersStatus] = useState<any>(null);
  const [validationResult, setValidationResult] = useState<any>(null);
  const [validating, setValidating] = useState<boolean>(false);

  const [geojsonFile, setGeojsonFile] = useState<File | null>(null);
  const [timeseriesFile, setTimeseriesFile] = useState<File | null>(null);
  const [labelsFile, setLabelsFile] = useState<File | null>(null);

  useEffect(() => {
    async function init() {
      try {
        const [schema, summary, providers] = await Promise.all([
          api.getSampleSchema(),
          api.getSummary(),
          api.getSatelliteProviders().catch(() => null)
        ]);
        setSchemaInfo(schema);
        setActiveDataset(summary);
        setProvidersStatus(providers);
      } catch (err) {
        console.error('Error initializing datasets page:', err);
      }
    }
    init();
  }, []);

  const handleValidate = async () => {
    if (!geojsonFile && !timeseriesFile && !labelsFile) {
      alert('Please select at least one file to validate.');
      return;
    }
    try {
      setValidating(true);
      const formData = new FormData();
      if (geojsonFile) formData.append('geojson_file', geojsonFile);
      if (timeseriesFile) formData.append('timeseries_file', timeseriesFile);
      if (labelsFile) formData.append('labels_file', labelsFile);

      const res = await api.validateDataset(formData);
      setValidationResult(res);
    } catch (err: any) {
      alert(`Validation failed: ${err.message}`);
    } finally {
      setValidating(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs">
        <h1 className="text-xl font-bold text-slate-900 flex items-center space-x-2">
          <Database className="w-5 h-5 text-emerald-600" />
          <span>Geospatial Data Ingestion & Schema Validation</span>
        </h1>
        <p className="text-xs text-slate-500 mt-0.5">
          Ingest Sentinel-2 optical feature tables, Sentinel-1 radar backscatter records, and GeoJSON field boundary polygons.
        </p>
      </div>

      {/* Active Dataset Provenance Card */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-3">
        <div className="flex flex-wrap items-center justify-between border-b border-slate-100 pb-2 gap-2">
          <h2 className="text-sm font-bold text-slate-900 flex items-center space-x-2">
            <Layers className="w-4 h-4 text-emerald-600" />
            <span>Currently Active Study Area</span>
          </h2>
          <span className="bg-emerald-100 text-emerald-800 font-semibold px-2.5 py-0.5 rounded text-xs">
            {activeDataset?.provenance || 'synthetic_demonstration'}
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 text-xs">
          <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
            <span className="text-slate-400 block text-[10px] uppercase font-semibold">Parcels Count</span>
            <strong className="text-base text-slate-900">{activeDataset?.total_parcels ?? 36}</strong>
          </div>
          <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
            <span className="text-slate-400 block text-[10px] uppercase font-semibold">Temporal Observations</span>
            <strong className="text-base text-slate-900">{activeDataset?.total_observations ?? 504}</strong>
          </div>
          <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
            <span className="text-slate-400 block text-[10px] uppercase font-semibold">Ground Truth Labels</span>
            <strong className="text-base text-slate-900">{activeDataset?.mean_actual_yield ? 'Available (36 fields)' : 'None'}</strong>
          </div>
          <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
            <span className="text-slate-400 block text-[10px] uppercase font-semibold">Observation Dates</span>
            <strong className="text-base text-slate-900">14 bi-weekly steps</strong>
          </div>
        </div>
      </div>

      {/* Satellite Providers & Real-Data Research Mode */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-3">
        <div className="flex flex-wrap items-center justify-between border-b border-slate-100 pb-2 gap-2">
          <h2 className="text-sm font-bold text-slate-900 flex items-center space-x-2">
            <Globe className="w-4 h-4 text-emerald-600" />
            <span>Real Satellite Data Providers & Research Mode</span>
          </h2>
          <span className={`px-2.5 py-0.5 rounded text-xs font-semibold ${
            providersStatus?.real_data_ready
              ? 'bg-emerald-100 text-emerald-800'
              : 'bg-amber-100 text-amber-800'
          }`}>
            {providersStatus?.real_data_ready ? 'Live Ingestion Authenticated' : 'Demonstration Mode (Synthetic)'}
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
          {/* Google Earth Engine */}
          <div className="bg-slate-50 p-3.5 rounded-lg border border-slate-200/70 space-y-1.5">
            <div className="flex items-center justify-between">
              <strong className="text-slate-900">Google Earth Engine</strong>
              <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${
                providersStatus?.providers?.google_earth_engine?.authenticated
                  ? 'bg-emerald-100 text-emerald-700'
                  : 'bg-slate-200 text-slate-600'
              }`}>
                {providersStatus?.providers?.google_earth_engine?.authenticated ? 'Authenticated' : 'No Credentials'}
              </span>
            </div>
            <p className="text-[11px] text-slate-500 leading-tight">
              {providersStatus?.providers?.google_earth_engine?.message || 'Requires EARTHENGINE_PROJECT or gcloud auth.'}
            </p>
          </div>

          {/* Copernicus Data Space */}
          <div className="bg-slate-50 p-3.5 rounded-lg border border-slate-200/70 space-y-1.5">
            <div className="flex items-center justify-between">
              <strong className="text-slate-900">Copernicus (CDSE)</strong>
              <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${
                providersStatus?.providers?.copernicus_data_space?.configured
                  ? 'bg-emerald-100 text-emerald-700'
                  : 'bg-slate-200 text-slate-600'
              }`}>
                {providersStatus?.providers?.copernicus_data_space?.configured ? 'Configured' : 'No Secret Set'}
              </span>
            </div>
            <p className="text-[11px] text-slate-500 leading-tight">
              {providersStatus?.providers?.copernicus_data_space?.message || 'Requires CDSE_CLIENT_ID and CDSE_CLIENT_SECRET.'}
            </p>
          </div>

          {/* STAC Catalog */}
          <div className="bg-slate-50 p-3.5 rounded-lg border border-slate-200/70 space-y-1.5">
            <div className="flex items-center justify-between">
              <strong className="text-slate-900">Public STAC Catalog</strong>
              <span className="text-[10px] px-1.5 py-0.5 rounded font-bold bg-emerald-100 text-emerald-700">
                Public API
              </span>
            </div>
            <p className="text-[11px] text-slate-500 leading-tight">
              Earth Search AWS open STAC endpoint accessible without cloud billing.
            </p>
          </div>
        </div>

        <div className="bg-slate-50/70 border border-slate-200 rounded-lg p-3 text-[11px] text-slate-600 flex items-start space-x-2">
          <ShieldCheck className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
          <span>
            <strong>Scientific Provenance Guarantee:</strong> CropYield-ST never synthesizes data and labels it as real satellite acquisitions. When external credentials are unavailable, the application operates honestly in Synthetic Demonstration Mode.
          </span>
        </div>
      </div>

      {/* Upload and Validation Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload Form */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-4">
          <h3 className="text-sm font-bold text-slate-900 border-b border-slate-100 pb-2 flex items-center space-x-2">
            <UploadCloud className="w-4 h-4 text-emerald-600" />
            <span>Upload New Field Dataset (Mode 2)</span>
          </h3>

          <div className="space-y-3 text-xs">
            {/* GeoJSON File */}
            <div className="space-y-1">
              <label className="font-semibold text-slate-700 block">
                1. Parcel Boundaries (GeoJSON Polygon Features)
              </label>
              <input
                type="file"
                accept=".geojson,.json"
                onChange={(e) => setGeojsonFile(e.target.files ? e.target.files[0] : null)}
                className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2 text-slate-700 file:mr-2 file:py-1 file:px-2 file:rounded file:border-0 file:text-xs file:bg-emerald-50 file:text-emerald-700 cursor-pointer"
              />
            </div>

            {/* Timeseries CSV File */}
            <div className="space-y-1">
              <label className="font-semibold text-slate-700 block">
                2. Multi-Sensor Time Series (CSV)
              </label>
              <input
                type="file"
                accept=".csv"
                onChange={(e) => setTimeseriesFile(e.target.files ? e.target.files[0] : null)}
                className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2 text-slate-700 file:mr-2 file:py-1 file:px-2 file:rounded file:border-0 file:text-xs file:bg-emerald-50 file:text-emerald-700 cursor-pointer"
              />
            </div>

            {/* Yield Labels CSV File */}
            <div className="space-y-1">
              <label className="font-semibold text-slate-700 block">
                3. Ground Truth Yield Labels (CSV, Optional for Prediction Mode)
              </label>
              <input
                type="file"
                accept=".csv"
                onChange={(e) => setLabelsFile(e.target.files ? e.target.files[0] : null)}
                className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2 text-slate-700 file:mr-2 file:py-1 file:px-2 file:rounded file:border-0 file:text-xs file:bg-emerald-50 file:text-emerald-700 cursor-pointer"
              />
            </div>

            <button
              onClick={handleValidate}
              disabled={validating}
              className={`w-full py-2.5 rounded-lg text-xs font-bold text-white flex items-center justify-center space-x-2 transition-colors cursor-pointer ${
                validating ? 'bg-slate-400 cursor-not-allowed' : 'bg-emerald-600 hover:bg-emerald-700 shadow-xs'
              }`}
            >
              {validating ? (
                <>
                  <Activity className="w-4 h-4 animate-spin" />
                  <span>Validating Dataset Schema...</span>
                </>
              ) : (
                <>
                  <CheckCircle className="w-4 h-4" />
                  <span>Validate Ingestion Schema</span>
                </>
              )}
            </button>
          </div>

          {/* Validation Report Feedback */}
          {validationResult && (
            <div
              className={`p-4 rounded-xl border text-xs space-y-2 ${
                validationResult.valid
                  ? 'bg-emerald-50 border-emerald-200 text-emerald-900'
                  : 'bg-red-50 border-red-200 text-red-900'
              }`}
            >
              <div className="font-bold flex items-center space-x-2">
                {validationResult.valid ? (
                  <CheckCircle className="w-4 h-4 text-emerald-600" />
                ) : (
                  <AlertTriangle className="w-4 h-4 text-red-600" />
                )}
                <span>
                  {validationResult.valid ? 'Validation Passed — Ready for Ingestion' : 'Validation Errors Encountered'}
                </span>
              </div>

              {validationResult.errors?.length > 0 && (
                <ul className="list-disc list-inside space-y-0.5 text-red-700">
                  {validationResult.errors.map((err: string, i: number) => (
                    <li key={i}>{err}</li>
                  ))}
                </ul>
              )}

              {validationResult.warnings?.length > 0 && (
                <ul className="list-disc list-inside space-y-0.5 text-amber-700">
                  {validationResult.warnings.map((warn: string, i: number) => (
                    <li key={i}>{warn}</li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </div>

        {/* Expected Schema Documentation */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-4">
          <h3 className="text-sm font-bold text-slate-900 border-b border-slate-100 pb-2 flex items-center space-x-2">
            <FileText className="w-4 h-4 text-emerald-600" />
            <span>Dataset Schema Specifications</span>
          </h3>

          <div className="space-y-3 text-xs text-slate-600">
            <div>
              <strong className="text-slate-900 block">Required Columns in Timeseries CSV:</strong>
              <div className="font-mono text-[11px] bg-slate-50 p-2 rounded border border-slate-200 mt-1">
                parcel_id, date, ndvi, evi, ndre, sar_vv_db, sar_vh_db, sar_rvi, optical_valid
              </div>
            </div>

            <div>
              <strong className="text-slate-900 block">Required Properties in GeoJSON:</strong>
              <div className="font-mono text-[11px] bg-slate-50 p-2 rounded border border-slate-200 mt-1">
                "properties": &#123; "parcel_id": "PARCEL_001", "crop_type": "Wheat", "area_ha": 12.5 &#125;
              </div>
            </div>

            <div>
              <strong className="text-slate-900 block">Ground Truth Labels CSV:</strong>
              <div className="font-mono text-[11px] bg-slate-50 p-2 rounded border border-slate-200 mt-1">
                parcel_id, yield_tha, crop_type
              </div>
            </div>

            <div className="bg-amber-50 border border-amber-200 p-3 rounded-lg text-amber-900 text-[11px] flex items-start space-x-2">
              <Info className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
              <span>
                <strong>Coordinate System:</strong> GeoJSON files must use WGS84 geographic coordinates (EPSG:4326) in longitude/latitude order.
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
