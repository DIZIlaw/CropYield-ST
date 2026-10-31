import {
  DashboardSummary,
  ParcelMetadata,
  ParcelDetail,
  ParcelTimeSeriesResponse,
  TrainRequest,
  TrainStatus,
  PredictRequest,
  PredictResponse,
  ExperimentRecord,
  AblationItem,
  AblationStudySummary,
  CloudRobustnessItem
} from '../types';

const API_BASE = (import.meta as any).env?.VITE_API_BASE || '/api';

export const api = {
  async getHealth() {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error('Failed to fetch health');
    return res.json();
  },

  async getSummary(): Promise<DashboardSummary> {
    const res = await fetch(`${API_BASE}/dashboard/summary`);
    if (!res.ok) throw new Error('Failed to fetch summary');
    return res.json();
  },

  async getParcels(): Promise<ParcelMetadata[]> {
    const res = await fetch(`${API_BASE}/parcels`);
    if (!res.ok) throw new Error('Failed to fetch parcels');
    return res.json();
  },

  async getParcelDetail(id: string): Promise<ParcelDetail> {
    const res = await fetch(`${API_BASE}/parcels/${id}`);
    if (!res.ok) throw new Error(`Failed to fetch parcel ${id}`);
    return res.json();
  },

  async getParcelGraph(): Promise<any> {
    const res = await fetch(`${API_BASE}/parcels/graph`);
    if (!res.ok) throw new Error('Failed to fetch parcel graph');
    return res.json();
  },

  async getParcelTimeSeries(id: string): Promise<ParcelTimeSeriesResponse> {
    const res = await fetch(`${API_BASE}/parcels/${id}/timeseries`);
    if (!res.ok) throw new Error(`Failed to fetch timeseries for parcel ${id}`);
    return res.json();
  },

  async predictYield(payload: PredictRequest): Promise<PredictResponse> {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Prediction failed');
    }
    return res.json();
  },

  async startTraining(payload: TrainRequest): Promise<{ job_id: string; message: string }> {
    const res = await fetch(`${API_BASE}/models/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Training failed to initiate');
    }
    return res.json();
  },

  async getTrainingStatus(jobId: string): Promise<TrainStatus> {
    const res = await fetch(`${API_BASE}/models/train/${jobId}`);
    if (!res.ok) throw new Error(`Failed to get status for job ${jobId}`);
    return res.json();
  },

  async getAvailableModels(): Promise<any[]> {
    const res = await fetch(`${API_BASE}/models`);
    if (!res.ok) throw new Error('Failed to fetch models list');
    return res.json();
  },

  async getExperiments(): Promise<ExperimentRecord[]> {
    const res = await fetch(`${API_BASE}/experiments`);
    if (!res.ok) throw new Error('Failed to fetch experiments');
    return res.json();
  },

  async getExperimentDetail(id: string): Promise<any> {
    const res = await fetch(`${API_BASE}/experiments/${id}`);
    if (!res.ok) throw new Error(`Failed to fetch experiment ${id}`);
    return res.json();
  },

  async getAblationStudy(): Promise<AblationStudySummary> {
    const res = await fetch(`${API_BASE}/evaluation/ablation`);
    if (!res.ok) throw new Error('Failed to fetch ablation results');
    return res.json();
  },

  async runAblationStudy(): Promise<AblationStudySummary> {
    const res = await fetch(`${API_BASE}/evaluation/ablation/run`, { method: 'POST' });
    if (!res.ok) throw new Error('Failed to run ablation study');
    return res.json();
  },

  async getCloudRobustness(): Promise<{ experiment_name: string; data: CloudRobustnessItem[] }> {
    const res = await fetch(`${API_BASE}/evaluation/cloud-robustness`);
    if (!res.ok) throw new Error('Failed to fetch cloud robustness analysis');
    return res.json();
  },

  async validateDataset(formData: FormData): Promise<any> {
    const res = await fetch(`${API_BASE}/datasets/validate`, {
      method: 'POST',
      body: formData
    });
    if (!res.ok) throw new Error('Validation request failed');
    return res.json();
  },

  async getSampleSchema(): Promise<any> {
    const res = await fetch(`${API_BASE}/datasets/sample-schema`);
    if (!res.ok) throw new Error('Failed to fetch sample schema');
    return res.json();
  },

  async getRegions(): Promise<any[]> {
    const res = await fetch(`${API_BASE}/datasets/regions`);
    if (!res.ok) throw new Error('Failed to fetch regional presets');
    return res.json();
  },

  async switchRegion(payload: { region_key?: string; custom_lat?: number; custom_lon?: number; region_name?: string }): Promise<any> {
    const res = await fetch(`${API_BASE}/datasets/switch-region`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Failed to switch region');
    }
    return res.json();
  },

  async getSatelliteProviders(): Promise<any> {
    const res = await fetch(`${API_BASE}/datasets/providers`);
    if (!res.ok) throw new Error('Failed to fetch satellite providers status');
    return res.json();
  }
};
