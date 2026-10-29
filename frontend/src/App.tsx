import React, { useState } from 'react';
import { Navbar } from './components/Navbar';
import { LocationSwitcherModal } from './components/LocationSwitcherModal';
import { OverviewDashboard } from './pages/OverviewDashboard';
import { GeospatialExplorer } from './pages/GeospatialExplorer';
import { SatelliteAnalytics } from './pages/SatelliteAnalytics';
import { YieldPrediction } from './pages/YieldPrediction';
import { ModelTraining } from './pages/ModelTraining';
import { ModelComparison } from './pages/ModelComparison';
import { DatasetManagement } from './pages/DatasetManagement';
import { ProjectDocumentation } from './pages/ProjectDocumentation';

export function App() {
  const [activeTab, setActiveTab] = useState<string>('overview');
  const [targetParcelId, setTargetParcelId] = useState<string | null>(null);
  const [isLocationModalOpen, setIsLocationModalOpen] = useState<boolean>(false);
  const [locationRefreshKey, setLocationRefreshKey] = useState<number>(0);

  const handleNavigate = (tab: string, parcelId?: string) => {
    if (parcelId) setTargetParcelId(parcelId);
    setActiveTab(tab);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleLocationChanged = () => {
    setTargetParcelId(null);
    setLocationRefreshKey((prev) => prev + 1);
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900 font-sans selection:bg-emerald-500 selection:text-white">
      {/* Top Navbar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={(tab) => {
          setActiveTab(tab);
          window.scrollTo({ top: 0, behavior: 'smooth' });
        }}
        isDemonstration={true}
        onOpenLocationModal={() => setIsLocationModalOpen(true)}
      />

      {/* Location Switcher Modal */}
      <LocationSwitcherModal
        isOpen={isLocationModalOpen}
        onClose={() => setIsLocationModalOpen(false)}
        onLocationChanged={handleLocationChanged}
      />

      {/* Main Content Surface */}
      <main key={locationRefreshKey} className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'overview' && <OverviewDashboard onNavigate={handleNavigate} />}
        {activeTab === 'explorer' && (
          <GeospatialExplorer
            initialParcelId={targetParcelId}
            onNavigateToAnalytics={(pid) => handleNavigate('analytics', pid)}
          />
        )}
        {activeTab === 'analytics' && <SatelliteAnalytics initialParcelId={targetParcelId} />}
        {activeTab === 'prediction' && <YieldPrediction initialParcelId={targetParcelId} />}
        {activeTab === 'training' && <ModelTraining />}
        {activeTab === 'ablation' && <ModelComparison />}
        {activeTab === 'datasets' && <DatasetManagement />}
        {activeTab === 'docs' && <ProjectDocumentation />}
      </main>

      {/* Academic Project Footer */}
      <footer className="bg-slate-900 border-t border-slate-800 text-slate-400 text-xs py-6 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-center sm:text-left">
          <div>
            <span className="font-bold text-white">CropYield-ST</span> — AI-Based Hyperlocal Crop Yield Forecasting
            <div className="text-slate-500 text-[11px] mt-0.5">
              Artificial Intelligence Course Project • B.Tech CSE, Fall Semester 2026–27
            </div>
          </div>
          <div className="text-[11px] text-slate-400">
            Aditya Pandey (24BCE2830) • Rishav Raj (24BCE2093) • Divyansh Singh (24BCE2821)
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
