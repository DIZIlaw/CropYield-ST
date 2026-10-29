export interface DashboardSummary {
  total_parcels: number;
  total_observations: number;
  latest_observation_date: string | null;
  trained_models_count: number;
  experiments_count: number;
  mean_predicted_yield: number | null;
  mean_actual_yield: number | null;
  provenance: string;
  is_demonstration: boolean;
  disclaimer: string;
}

export interface ParcelMetadata {
  parcel_id: string;
  crop_type: string;
  area_ha: number;
  soil_quality: number;
  centroid_lat: number;
  centroid_lon: number;
  neighbors_count: number;
  predicted_yield?: number | null;
  actual_yield?: number | null;
  geometry: {
    type: string;
    coordinates: any;
  };
}

export interface ParcelDetail extends ParcelMetadata {
  neighbors: string[];
  observation_count: number;
  optical_availability_pct: number;
  sar_availability_pct: number;
}

export interface TimeSeriesObservation {
  date: string;
  timestep: number;
  ndvi: number;
  evi?: number;
  ndre?: number;
  ndwi?: number;
  sar_vv_db: number;
  sar_vh_db: number;
  sar_vh_vv_ratio: number;
  sar_rvi: number;
  cloud_cover_percent: number;
  cloud_masked: boolean;
  optical_valid: boolean;
}

export interface ParcelTimeSeriesResponse {
  parcel_id: string;
  observations: TimeSeriesObservation[];
  provenance: string;
}

export interface TrainRequest {
  model_type: string;
  feature_set: string;
  epochs: number;
  learning_rate: number;
  hidden_dim: number;
  dropout: number;
  random_seed: number;
}

export interface TrainStatus {
  job_id: string;
  model_type: string;
  feature_set: string;
  epochs: number;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  result?: any;
  error?: string;
}

export interface PredictRequest {
  parcel_id: string;
  model_type: string;
  feature_set: string;
  crop_type?: string;
}

export interface ConformalUncertainty {
  interval_type: string;
  confidence_level: number;
  lower_bound: number;
  upper_bound: number;
  interval_width: number;
  method: string;
  is_validated: boolean;
  calibration_sample_size?: number;
  test_sample_size?: number;
  empirical_test_coverage_pct?: number | null;
  validation_status?: string;
  warnings?: string[];
}

export interface PredictResponse {
  parcel_id: string;
  predicted_yield_tha: number;
  actual_yield_tha?: number | null;
  label_type?: string;
  crop_type: string;
  unit?: string;
  model_used: string;
  model_version?: string;
  feature_set: string;
  latest_observation_date: string;
  data_availability: string;
  is_demonstration: boolean;
  provenance: string;
  neighbors: string[];
  confidence_interval: [number, number];
  uncertainty?: ConformalUncertainty;
  warnings?: string[];
}

export interface ExperimentRecord {
  experiment_id: string;
  model_type: string;
  feature_set: string;
  test_rmse: number | null;
  test_mae: number | null;
  test_r2: number | null;
  test_mape: number | null;
  duration_sec: number;
  provenance: string;
}

export interface AblationItem {
  name: string;
  model_type: string;
  feature_set: string;
  description: string;
  test_rmse: number;
  test_mae: number;
  test_r2: number;
  test_mape: number;
  duration_sec: number;
}

export interface AblationStudySummary {
  study_name: string;
  sample_count?: number;
  provenance?: string;
  is_demonstration?: boolean;
  evaluation_protocol?: string;
  split?: {
    train_samples: number;
    val_samples: number;
    test_samples: number;
  };
  results: AblationItem[];
}

export interface CloudRobustnessItem {
  cloud_occlusion_pct: number;
  optical_only_rmse: number;
  optical_only_r2: number;
  fusion_st_gnn_rmse: number;
  fusion_st_gnn_r2: number;
}
