import React from 'react';
import {
  LayoutDashboard,
  MapPin,
  LineChart,
  Target,
  Cpu,
  BarChart3,
  Database,
  BookOpen,
  Wheat,
  Activity,
  Globe
} from 'lucide-react';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  isDemonstration?: boolean;
  onOpenLocationModal?: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  isDemonstration = true,
  onOpenLocationModal
}) => {
  const navItems = [
    { id: 'overview', label: 'Overview', icon: LayoutDashboard },
    { id: 'explorer', label: 'Field Explorer', icon: MapPin },
    { id: 'analytics', label: 'Satellite Analytics', icon: LineChart },
    { id: 'prediction', label: 'Yield Prediction', icon: Target },
    { id: 'training', label: 'Model Training', icon: Cpu },
    { id: 'ablation', label: 'Ablation & Benchmarks', icon: BarChart3 },
    { id: 'datasets', label: 'Datasets', icon: Database },
    { id: 'docs', label: 'Project Docs', icon: BookOpen },
  ];

  return (
    <header className="bg-slate-900 border-b border-slate-800 text-white sticky top-0 z-50 shadow-md">
      {/* Top Banner with Academic Identity and Scientific Provenance */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-2.5 flex flex-wrap items-center justify-between border-b border-slate-800/80 gap-3">
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-lg bg-emerald-600 flex items-center justify-center text-white shadow-inner">
            <Wheat className="w-5 h-5 text-emerald-100" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-bold text-lg tracking-tight text-white">CropYield-ST</span>
              <span className="bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 text-xs px-2 py-0.5 rounded-full font-medium">
                ST-GNN v1.0
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Sentinel-1/2 SAR & Optical Fusion • Hyperlocal Yield Forecasting • B.Tech CSE 2026–27
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-3 text-xs">
          {/* Change Location Button */}
          {onOpenLocationModal && (
            <button
              onClick={onOpenLocationModal}
              className="flex items-center space-x-1.5 bg-emerald-600/30 hover:bg-emerald-600/50 border border-emerald-500/50 text-emerald-200 px-3 py-1 rounded-md transition-colors cursor-pointer"
              title="Click to view and analyze other agricultural locations"
            >
              <Globe className="w-3.5 h-3.5 text-emerald-400" />
              <span className="font-semibold">Change Location</span>
            </button>
          )}

          {/* Provenance Badge */}
          {isDemonstration ? (
            <div className="hidden sm:flex items-center space-x-1.5 bg-amber-500/10 border border-amber-500/30 text-amber-300 px-2.5 py-1 rounded-md">
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse"></span>
              <span className="font-medium">Demonstration Mode (Synthetic)</span>
            </div>
          ) : (
            <div className="hidden sm:flex items-center space-x-1.5 bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 px-2.5 py-1 rounded-md">
              <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
              <span className="font-medium">Live Satellite Ingestion Mode</span>
            </div>
          )}

          <div className="hidden md:flex items-center space-x-1 text-slate-400 border-l border-slate-700 pl-3">
            <Activity className="w-3.5 h-3.5 text-emerald-400" />
            <span>FastAPI: Active</span>
          </div>
        </div>
      </div>

      {/* Main Tab Navigation */}
      <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex space-x-1 overflow-x-auto py-1.5 no-scrollbar">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center space-x-2 px-3 py-2 rounded-md text-xs font-medium transition-colors whitespace-nowrap cursor-pointer ${
                  isActive
                    ? 'bg-emerald-600 text-white shadow-sm'
                    : 'text-slate-300 hover:text-white hover:bg-slate-800'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-white' : 'text-slate-400'}`} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>
      </nav>
    </header>
  );
};
