import React, { useState } from 'react';
import {
  Cpu,
  Play,
  Activity,
  CheckCircle2,
  AlertCircle,
  BarChart2,
  Sliders,
  Layers,
  Database
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { api } from '../services/api';
import { TrainRequest, TrainStatus } from '../types';

export const ModelTraining: React.FC = () => {
  const [modelType, setModelType] = useState<string>('st_gnn');
  const [featureSet, setFeatureSet] = useState<string>('fusion');
  const [epochs, setEpochs] = useState<number>(35);
  const [learningRate, setLearningRate] = useState<number>(0.005);
  const [hiddenDim, setHiddenDim] = useState<number>(64);
  const [dropout, setDropout] = useState<number>(0.2);
  const [randomSeed, setRandomSeed] = useState<number>(42);

  const [jobStatus, setJobStatus] = useState<TrainStatus | null>(null);
  const [trainingInProgress, setTrainingInProgress] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleStartTraining = async () => {
    try {
      setTrainingInProgress(true);
      setErrorMsg(null);
      setJobStatus(null);

      const payload: TrainRequest = {
        model_type: modelType,
        feature_set: featureSet,
        epochs: Number(epochs),
        learning_rate: Number(learningRate),
        hidden_dim: Number(hiddenDim),
        dropout: Number(dropout),
        random_seed: Number(randomSeed)
      };

      const startRes = await api.startTraining(payload);
      const jobId = startRes.job_id;

      // Poll until finished
      const pollInterval = setInterval(async () => {
        try {
          const status = await api.getTrainingStatus(jobId);
          setJobStatus(status);
          if (status.status === 'completed' || status.status === 'failed') {
            clearInterval(pollInterval);
            setTrainingInProgress(false);
            if (status.status === 'failed') {
              setErrorMsg(status.error || 'Training failed');
            }
          }
        } catch (pollErr) {
          clearInterval(pollInterval);
          setTrainingInProgress(false);
        }
      }, 800);
    } catch (err: any) {
      setTrainingInProgress(false);
      setErrorMsg(err.message || 'Failed to start training');
    }
  };

  const lossData = jobStatus?.result?.loss_history || [];
  const testMetrics = jobStatus?.result?.test_metrics;

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs">
        <h1 className="text-xl font-bold text-slate-900 flex items-center space-x-2">
          <Cpu className="w-5 h-5 text-emerald-600" />
          <span>Deep Learning Model Training & Hyperparameter Tuning</span>
        </h1>
        <p className="text-xs text-slate-500 mt-0.5">
          Train Spatio-Temporal Graph Neural Networks with spatial block validation to prevent spatial data leakage.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Training Configurations */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-4">
          <h2 className="text-sm font-bold text-slate-900 border-b border-slate-100 pb-3 flex items-center space-x-2">
            <Sliders className="w-4 h-4 text-emerald-600" />
            <span>Training Configuration</span>
          </h2>

          <div className="space-y-3 text-xs">
            {/* Model Architecture */}
            <div className="space-y-1">
              <label className="font-semibold text-slate-700">Model Architecture</label>
              <select
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
                className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2 text-slate-800 focus:ring-2 focus:ring-emerald-500 cursor-pointer"
              >
                <option value="st_gnn">CropYield-ST (Spatio-Temporal GNN)</option>
                <option value="pure_temporal_rnn">Pure Temporal GRU (No Graph)</option>
                <option value="xgboost">XGBoost (Tabular Baseline)</option>
              </select>
            </div>

            {/* Feature Set */}
            <div className="space-y-1">
              <label className="font-semibold text-slate-700">Feature Ingestion Set</label>
              <select
                value={featureSet}
                onChange={(e) => setFeatureSet(e.target.value)}
                className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2 text-slate-800 focus:ring-2 focus:ring-emerald-500 cursor-pointer"
              >
                <option value="fusion">Optical + SAR Dual-Band Fusion</option>
                <option value="optical_only">Optical Features Only (NDVI, NDRE, EVI)</option>
                <option value="sar_only">SAR C-Band Only (VV, VH, RVI)</option>
              </select>
            </div>

            {/* Epochs */}
            <div className="space-y-1">
              <div className="flex justify-between font-semibold text-slate-700">
                <span>Epochs:</span>
                <span className="text-emerald-700 font-bold">{epochs}</span>
              </div>
              <input
                type="range"
                min="10"
                max="100"
                step="5"
                value={epochs}
                onChange={(e) => setEpochs(Number(e.target.value))}
                className="w-full accent-emerald-600 cursor-pointer"
              />
            </div>

            {/* Learning Rate */}
            <div className="space-y-1">
              <label className="font-semibold text-slate-700">Learning Rate</label>
              <select
                value={learningRate}
                onChange={(e) => setLearningRate(Number(e.target.value))}
                className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2 text-slate-800 focus:ring-2 focus:ring-emerald-500 cursor-pointer"
              >
                <option value={0.01}>0.01 (Aggressive)</option>
                <option value={0.005}>0.005 (Standard AdamW)</option>
                <option value={0.001}>0.001 (Conservative)</option>
              </select>
            </div>

            {/* Hidden Dimensions */}
            <div className="space-y-1">
              <label className="font-semibold text-slate-700">Hidden Embedding Dimension</label>
              <select
                value={hiddenDim}
                onChange={(e) => setHiddenDim(Number(e.target.value))}
                className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2 text-slate-800 focus:ring-2 focus:ring-emerald-500 cursor-pointer"
              >
                <option value={32}>32 units</option>
                <option value={64}>64 units (Optimal)</option>
                <option value={128}>128 units</option>
              </select>
            </div>

            {/* Dropout */}
            <div className="space-y-1">
              <div className="flex justify-between font-semibold text-slate-700">
                <span>Dropout Regularization:</span>
                <span className="text-emerald-700 font-bold">{dropout}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="0.5"
                step="0.05"
                value={dropout}
                onChange={(e) => setDropout(Number(e.target.value))}
                className="w-full accent-emerald-600 cursor-pointer"
              />
            </div>

            {/* Random Seed */}
            <div className="space-y-1">
              <label className="font-semibold text-slate-700">Reproducibility Seed</label>
              <input
                type="number"
                value={randomSeed}
                onChange={(e) => setRandomSeed(Number(e.target.value))}
                className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2 text-slate-800 focus:ring-2 focus:ring-emerald-500"
              />
            </div>
          </div>

          <button
            onClick={handleStartTraining}
            disabled={trainingInProgress}
            className={`w-full py-3 rounded-lg text-xs font-bold text-white flex items-center justify-center space-x-2 transition-colors cursor-pointer ${
              trainingInProgress ? 'bg-slate-400 cursor-not-allowed' : 'bg-emerald-600 hover:bg-emerald-700 shadow-sm'
            }`}
          >
            {trainingInProgress ? (
              <>
                <Activity className="w-4 h-4 animate-spin" />
                <span>Training in Progress...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-white" />
                <span>Execute Training Job</span>
              </>
            )}
          </button>

          {errorMsg && (
            <div className="bg-red-50 border border-red-200 text-red-700 p-3 rounded-lg text-xs flex items-start space-x-2">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
              <span>{errorMsg}</span>
            </div>
          )}
        </div>

        {/* Right 2 Columns: Live Loss Curves & Validation Results */}
        <div className="lg:col-span-2 space-y-6">
          {/* Training Loss Curve */}
          <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-4">
            <div className="flex flex-wrap items-center justify-between border-b border-slate-100 pb-3 gap-2">
              <div>
                <h3 className="text-sm font-bold text-slate-900 flex items-center space-x-2">
                  <BarChart2 className="w-4 h-4 text-emerald-600" />
                  <span>Training & Validation Convergence Curves</span>
                </h3>
                <p className="text-xs text-slate-500">
                  Mean Squared Error loss per epoch across spatially separated field blocks.
                </p>
              </div>

              {jobStatus?.status === 'completed' && (
                <div className="flex items-center space-x-1.5 text-xs text-emerald-700 bg-emerald-50 px-2.5 py-1 rounded-md border border-emerald-200 font-semibold">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  <span>Converged in {jobStatus.result?.duration_sec}s</span>
                </div>
              )}
            </div>

            {lossData.length > 0 ? (
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={lossData} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} />
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
                      dataKey="train_loss"
                      name="Train Loss (MSE)"
                      stroke="#059669"
                      strokeWidth={2}
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="val_loss"
                      name="Validation Loss (Spatial Block)"
                      stroke="#2563eb"
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="h-64 flex items-center justify-center border border-dashed border-slate-200 rounded-lg text-xs text-slate-400">
                Execute a training run to visualize live loss convergence curves.
              </div>
            )}
          </div>

          {/* Validation & Test Benchmark Metrics Card */}
          {testMetrics && (
            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-4">
              <h3 className="text-sm font-bold text-slate-900 border-b border-slate-100 pb-2">
                Out-of-Fold Evaluation Metrics (Held-Out Test Fields)
              </h3>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
                  <span className="text-[10px] text-slate-500 uppercase font-bold block">RMSE</span>
                  <div className="text-xl font-extrabold text-slate-900 mt-0.5">{testMetrics.rmse}</div>
                  <span className="text-[10px] text-slate-400">tonnes/hectare</span>
                </div>

                <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
                  <span className="text-[10px] text-slate-500 uppercase font-bold block">MAE</span>
                  <div className="text-xl font-extrabold text-slate-900 mt-0.5">{testMetrics.mae}</div>
                  <span className="text-[10px] text-slate-400">tonnes/hectare</span>
                </div>

                <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
                  <span className="text-[10px] text-slate-500 uppercase font-bold block">R² Score</span>
                  <div className="text-xl font-extrabold text-emerald-700 mt-0.5">{testMetrics.r2}</div>
                  <span className="text-[10px] text-slate-400">Variance explained</span>
                </div>

                <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
                  <span className="text-[10px] text-slate-500 uppercase font-bold block">MAPE</span>
                  <div className="text-xl font-extrabold text-slate-900 mt-0.5">{testMetrics.mape_percent}%</div>
                  <span className="text-[10px] text-slate-400">Percentage error</span>
                </div>
              </div>

              {/* Spatial Partition Summary */}
              {jobStatus?.result?.split && (
                <div className="text-xs text-slate-500 bg-slate-50 p-3 rounded-lg border border-slate-200/60 flex items-center justify-between">
                  <span>
                    Spatial Partition: <strong>{jobStatus.result.split.train_count}</strong> train fields •{' '}
                    <strong>{jobStatus.result.split.val_count}</strong> val fields •{' '}
                    <strong>{jobStatus.result.split.test_count}</strong> test fields
                  </span>
                  <span className="text-emerald-700 font-semibold">Zero Spatial Leakage</span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
