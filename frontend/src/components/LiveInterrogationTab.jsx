import React, { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine
} from 'recharts';
import { Search, Loader2, Info, CheckCircle2, AlertTriangle, Settings, Sparkles } from 'lucide-react';
import { formatPercent } from '../utils/format';

export default function LiveInterrogationTab({ health }) {
  const [text, setText] = useState('');
  const [modelChoice, setModelChoice] = useState('DistilBERT (Transformer)');
  const [modelDir, setModelDir] = useState('./bert_fake_reviews_model');
  const [fusionDir, setFusionDir] = useState('./fusion_model');
  const [showSettings, setShowSettings] = useState(false);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
  const isButtonDisabled = wordCount < 3 || loading;

  const handleAnalyze = async () => {
    if (wordCount < 3) return;
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          model_choice: modelChoice,
          model_dir: modelDir,
          fusion_dir: fusionDir,
        }),
      });

      let data;
      try {
        data = await response.json();
      } catch (parseErr) {
        throw new Error(`Server returned status ${response.status} (${response.statusText}). Check backend logs.`);
      }
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to get analysis response from server.');
      }
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const sampleReviews = [
    {
      label: 'Real Sample',
      text: 'I bought this for my kitchen and it works great, the build quality is solid and it has lasted me over a year without any issues at all.',
    },
    {
      label: 'Fake Sample',
      text: 'This product is amazing and exceptional. Very nice design and highly recommend to all buyers looking for standard high quality results.',
    },
  ];

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-indigo-50 border border-indigo-100 text-indigo-700 text-xs font-semibold mb-2">
            <Search className="w-3.5 h-3.5 text-indigo-600" />
            LIVE MODEL INTERROGATION
          </div>
          <h2 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
            Interrogate any product review
          </h2>
          <p className="text-sm text-slate-500 mt-1">
            Paste a product review below and select which trained model classifies its authenticity.
          </p>
        </div>

        <button
          onClick={() => setShowSettings(!showSettings)}
          className="self-start sm:self-auto inline-flex items-center gap-2 bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 text-xs font-mono px-3 py-2 rounded-lg transition-colors shadow-2xs cursor-pointer"
        >
          <Settings className="w-3.5 h-3.5 text-slate-500" />
          {showSettings ? 'Hide Path Config' : 'Model Paths'}
        </button>
      </div>

      {/* Path Settings Accordion */}
      {showSettings && (
        <div className="bg-slate-50 border border-slate-200 rounded-xl p-5 space-y-4 animate-fadeIn">
          <h4 className="text-xs font-mono uppercase text-slate-600 font-bold">⚙️ Custom Model Directory Paths</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-slate-600 font-mono mb-1">DistilBERT model folder path</label>
              <input
                type="text"
                value={modelDir}
                onChange={(e) => setModelDir(e.target.value)}
                className="w-full bg-white border border-slate-300 rounded-lg px-3 py-2 text-xs font-mono text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              />
            </div>
            <div>
              <label className="block text-xs text-slate-600 font-mono mb-1">Fusion model folder path</label>
              <input
                type="text"
                value={fusionDir}
                onChange={(e) => setFusionDir(e.target.value)}
                className="w-full bg-white border border-slate-300 rounded-lg px-3 py-2 text-xs font-mono text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              />
            </div>
          </div>
        </div>
      )}

      {/* Input Form */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-5 shadow-2xs">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="w-full sm:w-80">
            <label className="block text-xs font-mono text-slate-500 mb-1">Select Classification Model</label>
            <select
              value={modelChoice}
              onChange={(e) => setModelChoice(e.target.value)}
              className="w-full bg-white border border-slate-300 rounded-lg px-3 py-2 text-sm font-semibold text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            >
              <option value="DistilBERT (Transformer)">DistilBERT (Transformer)</option>
              <option value="Fusion - Logistic Regression">Fusion - Logistic Regression</option>
              <option value="Fusion - Linear SVM">Fusion - Linear SVM</option>
            </select>
          </div>

          {/* Quick Preset Buttons */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500 font-mono">Quick test:</span>
            {sampleReviews.map((sample, idx) => (
              <button
                key={idx}
                onClick={() => setText(sample.text)}
                className="text-xs font-mono bg-slate-50 border border-slate-300 hover:bg-slate-100 text-slate-700 px-3 py-1.5 rounded-md transition-all cursor-pointer"
              >
                {sample.label}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="block text-xs font-mono text-slate-500 mb-1">Review Text Input</label>
          <textarea
            rows={5}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="e.g. I bought this for my kitchen and it works great, the build quality is solid and it has lasted me over a year without any issues at all."
            className="w-full bg-white border border-slate-300 rounded-xl p-4 text-sm text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all shadow-2xs"
          ></textarea>
        </div>

        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pt-1">
          <span className="text-xs font-mono text-slate-500">
            {wordCount} words {wordCount < 3 && '(minimum 3 required)'}
          </span>

          <button
            onClick={handleAnalyze}
            disabled={isButtonDisabled}
            className={`inline-flex items-center justify-center gap-2 px-6 py-2.5 rounded-lg font-semibold text-sm transition-all shadow-sm ${
              isButtonDisabled
                ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                : 'bg-indigo-600 hover:bg-indigo-700 text-white cursor-pointer'
            }`}
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin text-white" />
                Running Analysis...
              </>
            ) : (
              <>
                <Search className="w-4 h-4" />
                Analyze Review
              </>
            )}
          </button>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="bg-rose-50 border border-rose-200 rounded-xl p-4 flex items-center gap-3 text-sm text-rose-700 font-mono">
          <AlertTriangle className="w-5 h-5 shrink-0 text-rose-600" />
          <span>{error}</span>
        </div>
      )}

      {/* Output Results */}
      {result && (
        <div className="space-y-6 animate-fadeIn">
          {/* Verdict Card */}
          <div
            className={`rounded-xl p-6 border transition-all shadow-2xs ${
              result.label === 'real'
                ? 'bg-emerald-50/90 border-emerald-200 text-emerald-900'
                : 'bg-rose-50/90 border-rose-200 text-rose-900'
            }`}
          >
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
              <div>
                <div
                  className={`font-mono text-xl sm:text-2xl font-bold flex items-center gap-2 ${
                    result.label === 'real' ? 'text-emerald-700' : 'text-rose-700'
                  }`}
                >
                  {result.label === 'real' ? (
                    <>
                      <CheckCircle2 className="w-7 h-7 text-emerald-600" />
                      ✓ LIKELY GENUINE
                    </>
                  ) : (
                    <>
                      <AlertTriangle className="w-7 h-7 text-rose-600" />
                      ⚑ LIKELY AI-GENERATED
                    </>
                  )}
                </div>
                <div className="text-xs sm:text-sm text-slate-600 font-mono mt-1">
                  Model confidence: <strong className="text-slate-900">{formatPercent(result.confidence)}</strong>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <span className="text-xs font-mono text-slate-600 bg-white border border-slate-200 px-3 py-1.5 rounded-lg shadow-2xs">
                  Device: {result.device}
                </span>
              </div>
            </div>
          </div>

          {/* Probability Meters */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-2xs">
              <div className="flex justify-between items-center mb-2 font-mono">
                <span className="text-xs text-slate-500 font-semibold">Real Probability</span>
                <span className="text-lg font-bold text-emerald-600">{formatPercent(result.prob_real)}</span>
              </div>
              <div className="w-full bg-slate-100 h-3 rounded-full overflow-hidden">
                <div
                  className="bg-emerald-600 h-full transition-all duration-500"
                  style={{ width: `${result.prob_real * 100}%` }}
                ></div>
              </div>
            </div>

            <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-2xs">
              <div className="flex justify-between items-center mb-2 font-mono">
                <span className="text-xs text-slate-500 font-semibold">Fake Probability</span>
                <span className="text-lg font-bold text-rose-600">{formatPercent(result.prob_fake)}</span>
              </div>
              <div className="w-full bg-slate-100 h-3 rounded-full overflow-hidden">
                <div
                  className="bg-rose-600 h-full transition-all duration-500"
                  style={{ width: `${result.prob_fake * 100}%` }}
                ></div>
              </div>
            </div>
          </div>

          {/* SHAP Explanation Chart */}
          <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-6 shadow-2xs">
            <div>
              <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-indigo-600" />
                🔬 Why did the model make this prediction?
              </h3>
              <p className="text-xs sm:text-sm text-slate-500 mt-1">
                Which specific words/features pushed this review toward FAKE vs. REAL
              </p>
            </div>

            {result.contributions && result.contributions.length > 0 ? (
              <div className="w-full h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={result.contributions}
                    layout="vertical"
                    margin={{ top: 10, right: 40, left: 60, bottom: 20 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                    <XAxis type="number" stroke="#64748B" tick={{ fill: '#64748B', fontSize: 11 }} />
                    <YAxis dataKey="display_token" type="category" stroke="#64748B" tick={{ fill: '#0F172A', fontSize: 12, fontWeight: 500 }} width={90} />
                    <ReferenceLine x={0} stroke="#94A3B8" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#FFFFFF',
                        borderColor: '#E2E8F0',
                        color: '#0F172A',
                        borderRadius: '8px',
                        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                        fontSize: '12px',
                      }}
                      formatter={(val) => [val, 'Contribution']}
                    />
                    <Bar dataKey="shap_value" radius={[2, 2, 2, 2]}>
                      {result.contributions.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.shap_value > 0 ? '#E11D48' : '#059669'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-6 text-center text-xs text-slate-500 font-mono">
                No strong word/feature contributions extracted for this text.
              </div>
            )}

            {/* Chart Legend Box */}
            <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-xs text-slate-600 space-y-1.5">
              <div className="font-mono text-sm font-bold text-slate-900">🔬 How to read this chart</div>
              <p>
                <span className="text-rose-600 font-bold">Rose bars</span> push the prediction toward <b>FAKE</b>.{' '}
                <span className="text-emerald-600 font-bold">Green bars</span> push it toward <b>REAL</b>.
              </p>
              <p>
                This color coding is consistent across predictions. Longer bars indicate stronger influence on this review's outcome.
              </p>
            </div>

            {/* Top Influential Table */}
            {result.top_influential && result.top_influential.length > 0 && (
              <div className="space-y-3 pt-4">
                <h4 className="text-sm font-bold text-slate-900">🔍 Most influential words / features</h4>
                <div className="overflow-x-auto border border-slate-200 rounded-lg shadow-2xs">
                  <table className="w-full text-left text-xs font-mono">
                    <thead className="bg-slate-50 text-slate-600 border-b border-slate-200">
                      <tr>
                        <th className="p-3">Word / Feature</th>
                        <th className="p-3">Contribution</th>
                        <th className="p-3">Pushes toward</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 text-slate-800">
                      {result.top_influential.map((row, idx) => (
                        <tr key={idx} className="hover:bg-slate-50">
                          <td className="p-3 font-semibold text-slate-900">{row.token}</td>
                          <td className="p-3">{row.contribution}</td>
                          <td className={`p-3 font-bold ${row.pushes_toward.includes('FAKE') ? 'text-rose-600' : 'text-emerald-600'}`}>
                            {row.pushes_toward}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
