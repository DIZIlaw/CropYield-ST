import React, { useMemo, useEffect } from 'react';
import { MapContainer, TileLayer, Polygon as LeafletPolygon, Popup, Tooltip, useMap } from 'react-leaflet';
import { ParcelMetadata } from '../types';

interface ParcelMapProps {
  parcels: ParcelMetadata[];
  selectedParcelId: string | null;
  onSelectParcel: (parcelId: string) => void;
  neighborIds?: string[];
  colorMode?: 'yield' | 'ndvi' | 'soil' | 'default';
}

function ChangeMapView({ center }: { center: [number, number] }) {
  const map = useMap();
  useEffect(() => {
    map.flyTo(center, 14, { duration: 1.2 });
  }, [center, map]);
  return null;
}

export const ParcelMap: React.FC<ParcelMapProps> = ({
  parcels,
  selectedParcelId,
  onSelectParcel,
  neighborIds = [],
  colorMode = 'yield'
}) => {
  // Center map on the centroid of the study area
  const center: [number, number] = useMemo(() => {
    if (parcels.length === 0) return [30.9100, 75.8700];
    const avgLat = parcels.reduce((sum, p) => sum + p.centroid_lat, 0) / parcels.length;
    const avgLon = parcels.reduce((sum, p) => sum + p.centroid_lon, 0) / parcels.length;
    return [avgLat, avgLon];
  }, [parcels]);

  // Color interpolation helper for Yield (tonnes/ha)
  const getYieldColor = (yieldVal?: number | null) => {
    if (yieldVal === undefined || yieldVal === null) return '#94a3b8'; // slate-400
    if (yieldVal < 2.5) return '#facc15'; // yellow-400
    if (yieldVal < 3.8) return '#a3e635'; // lime-400
    if (yieldVal < 5.0) return '#4ade80'; // green-400
    if (yieldVal < 6.0) return '#10b981'; // emerald-500
    return '#047857'; // emerald-700
  };

  const getSoilColor = (soilVal?: number | null) => {
    if (!soilVal) return '#94a3b8';
    if (soilVal < 0.6) return '#fed7aa';
    if (soilVal < 0.8) return '#fdba74';
    return '#ea580c';
  };

  return (
    <div className="relative w-full h-[520px] rounded-xl overflow-hidden border border-slate-200 shadow-xs bg-slate-100">
      <MapContainer
        center={center}
        zoom={14}
        scrollWheelZoom={true}
        className="w-full h-full"
      >
        <ChangeMapView center={center} />
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {parcels.map((parcel) => {
          const isSelected = parcel.parcel_id === selectedParcelId;
          const isNeighbor = neighborIds.includes(parcel.parcel_id);

          // Convert GeoJSON coordinates [lon, lat] -> Leaflet [lat, lon]
          const coords = parcel.geometry.coordinates[0].map((pt: [number, number]) => [pt[1], pt[0]] as [number, number]);

          let fillColor = '#10b981';
          if (colorMode === 'yield') {
            fillColor = getYieldColor(parcel.predicted_yield);
          } else if (colorMode === 'soil') {
            fillColor = getSoilColor(parcel.soil_quality);
          }

          let strokeColor = '#065f46';
          let weight = 1.5;
          let dashArray = '';
          let fillOpacity = 0.55;

          if (isSelected) {
            strokeColor = '#0284c7'; // blue-600 focus
            weight = 3.5;
            fillOpacity = 0.85;
          } else if (isNeighbor) {
            strokeColor = '#d97706'; // amber-600
            weight = 2.5;
            dashArray = '5, 5';
            fillOpacity = 0.70;
          }

          return (
            <LeafletPolygon
              key={parcel.parcel_id}
              positions={coords}
              pathOptions={{
                color: strokeColor,
                weight: weight,
                fillColor: fillColor,
                fillOpacity: fillOpacity,
                dashArray: dashArray,
              }}
              eventHandlers={{
                click: () => onSelectParcel(parcel.parcel_id)
              }}
            >
              <Tooltip sticky>
                <div className="text-xs font-sans">
                  <div className="font-bold text-slate-900">{parcel.parcel_id}</div>
                  <div className="text-slate-600">{parcel.crop_type}</div>
                  <div className="text-emerald-700 font-semibold mt-0.5">
                    Forecast: {parcel.predicted_yield ? `${parcel.predicted_yield} t/ha` : 'Not run'}
                  </div>
                  {isNeighbor && (
                    <div className="text-amber-700 text-[10px] font-medium mt-0.5">
                      ★ Adjacent Neighbor Field
                    </div>
                  )}
                </div>
              </Tooltip>
              <Popup>
                <div className="text-xs p-1">
                  <h4 className="font-bold text-sm text-slate-900">{parcel.parcel_id}</h4>
                  <p className="text-slate-500 mb-2">{parcel.crop_type} • {parcel.area_ha} ha</p>
                  <div className="space-y-1 border-t border-slate-200 pt-1.5">
                    <div className="flex justify-between">
                      <span className="text-slate-500">Predicted Yield:</span>
                      <span className="font-bold text-emerald-700">{parcel.predicted_yield ?? 'N/A'} t/ha</span>
                    </div>
                    {parcel.actual_yield && (
                      <div className="flex justify-between">
                        <span className="text-slate-500">Actual Label:</span>
                        <span className="font-medium text-slate-800">{parcel.actual_yield} t/ha</span>
                      </div>
                    )}
                    <div className="flex justify-between">
                      <span className="text-slate-500">Soil Quality:</span>
                      <span className="font-medium text-slate-800">{(parcel.soil_quality * 100).toFixed(0)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">Adjacency Degree:</span>
                      <span className="font-medium text-slate-800">{parcel.neighbors_count} neighbours</span>
                    </div>
                  </div>
                  <button
                    onClick={() => onSelectParcel(parcel.parcel_id)}
                    className="w-full mt-2.5 bg-emerald-600 hover:bg-emerald-700 text-white py-1 px-2 rounded text-xs font-medium cursor-pointer"
                  >
                    Inspect Field Details
                  </button>
                </div>
              </Popup>
            </LeafletPolygon>
          );
        })}
      </MapContainer>

      {/* Floating Map Legend */}
      <div className="absolute bottom-4 right-4 bg-white/95 backdrop-blur-xs border border-slate-300 rounded-lg p-3 text-xs shadow-md z-1000 pointer-events-auto">
        <h5 className="font-semibold text-slate-900 mb-1.5">
          {colorMode === 'yield' ? 'Yield Forecast (t/ha)' : 'Soil Index'}
        </h5>
        {colorMode === 'yield' ? (
          <div className="space-y-1">
            <div className="flex items-center space-x-2">
              <span className="w-3.5 h-3.5 rounded-xs" style={{ backgroundColor: '#047857' }}></span>
              <span className="text-slate-700">&gt; 6.0 t/ha (High)</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="w-3.5 h-3.5 rounded-xs" style={{ backgroundColor: '#10b981' }}></span>
              <span className="text-slate-700">5.0 - 6.0 t/ha</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="w-3.5 h-3.5 rounded-xs" style={{ backgroundColor: '#4ade80' }}></span>
              <span className="text-slate-700">3.8 - 5.0 t/ha</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="w-3.5 h-3.5 rounded-xs" style={{ backgroundColor: '#facc15' }}></span>
              <span className="text-slate-700">&lt; 3.8 t/ha (Moderate/Mustard)</span>
            </div>
          </div>
        ) : (
          <div className="space-y-1">
            <div className="flex items-center space-x-2">
              <span className="w-3.5 h-3.5 rounded-xs bg-amber-600"></span>
              <span className="text-slate-700">High Fertility</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="w-3.5 h-3.5 rounded-xs bg-amber-400"></span>
              <span className="text-slate-700">Moderate Fertility</span>
            </div>
          </div>
        )}

        <div className="border-t border-slate-200 mt-2 pt-2 space-y-1">
          <div className="flex items-center space-x-2">
            <span className="w-3.5 h-1 border-b-2 border-blue-600"></span>
            <span className="text-slate-600">Selected Parcel</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="w-3.5 h-1 border-b-2 border-dashed border-amber-600"></span>
            <span className="text-slate-600">Adjacency Neighbors</span>
          </div>
        </div>
      </div>
    </div>
  );
};
