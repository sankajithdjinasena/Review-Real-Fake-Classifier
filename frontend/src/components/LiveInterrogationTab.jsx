import React, { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine
} from 'recharts';
import { Search, Loader2, Info, CheckCircle2, AlertTriangle, Settings, Sparkles } from 'lucide-react';

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
      label: "Real Sample",
      text: "I bought this for my kitchen and it works great, the build quality is solid and it has lasted me over a year without any issues at all."
    },
    {
      label: "Fake Sample",
      text: "This product is amazing and exceptional. Very nice design and highly recommend to all buyers looking for standard high quality results."
    }
  ];

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <div className="inline-flex items-center gap-2 font-mono text-xs text-[#8DA0A8] border border-[#2E3A41] px-2.5 py-1 rounded bg-[#1B2328] mb-2">
            <span className="w-2 h-2 rounded-full bg-[#5FD3A0]"></span>
            LIVE MODEL INTERROGATION
          </div>
          <h2 className="text-2xl sm:text-3xl font-extrabold text-[#E9EDEE]">
            Try it yourself
          </h2>
          <p className="text-sm text-[#8DA0A8] mt-1">
            Paste a product review below and pick which model classifies it.
          </p>
        </div>

        <button
          onClick={() => setShowSettings(!showSettings)}
          className="self-start sm:self-auto inline-flex items-center gap-2 bg-[#1B2328] border border-[#2E3A41] text-[#8DA0A8] hover:text-[#E9EDEE] text-xs font-mono px-3 py-2 rounded-lg transition-colors"
        >
          <Settings className="w-4 h-4" />
          {showSettings ? 'Hide Path Config' : 'Model Paths'}
        </button>
      </div>

      {/* Path Settings Accordion */}
      {showSettings && (
        <div className="bg-[#1B2328] border border-[#2E3A41] rounded-xl p-5 space-y-4 animate-fadeIn">
          <h4 className="text-xs font-mono uppercase text-[#8DA0A8] font-bold">⚙️ Custom Model Directory Paths</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-[#8DA0A8] font-mono mb-1">DistilBERT model folder path</label>
              <input
                type="text"
                value={modelDir}
                onChange={(e) => setModelDir(e.target.value)}
                className="w-full bg-[#212B31] border border-[#2E3A41] rounded-lg px-3 py-2 text-xs font-mono text-[#E9EDEE] focus:outline-none focus:border-[#5FD3A0]"
              />
            </div>
            <div>
              <label className="block text-xs text-[#8DA0A8] font-mono mb-1">Fusion model folder path</label>
              <input
                type="text"
                value={fusionDir}
                onChange={(e) => setFusionDir(e.target.value)}
                className="w-full bg-[#212B31] border border-[#2E3A41] rounded-lg px-3 py-2 text-xs font-mono text-[#E9EDEE] focus:outline-none focus:border-[#5FD3A0]"
              />
            </div>
          </div>
        </div>
      )}

      {/* Input Form */}
      <div className="bg-[#1B2328] border border-[#2E3A41] rounded-xl p-6 space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="w-full sm:w-80">
            <label className="block text-xs font-mono text-[#8DA0A8] mb-1">Select Model</label>
            <select
              value={modelChoice}
              onChange={(e) => setModelChoice(e.target.value)}
              className="w-full bg-[#212B31] border border-[#2E3A41] rounded-lg px-3 py-2.5 text-sm font-semibold text-[#E9EDEE] focus:outline-none focus:border-[#5FD3A0]"
            >
              <option value="DistilBERT (Transformer)">DistilBERT (Transformer)</option>
              <option value="Fusion - Logistic Regression">Fusion - Logistic Regression</option>
              <option value="Fusion - Linear SVM">Fusion - Linear SVM</option>
            </select>
          </div>

          {/* Quick Preset Buttons */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-[#8DA0A8] font-mono">Quick test:</span>
            {sampleReviews.map((sample, idx) => (
              <button
                key={idx}
                onClick={() => setText(sample.text)}
                className="text-xs font-mono bg-[#212B31] border border-[#2E3A41] hover:border-[#8DA0A8] text-[#E9EDEE] px-2.5 py-1.5 rounded transition-all"
              >
                {sample.label}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="block text-xs font-mono text-[#8DA0A8] mb-1">Review text</label>
          <textarea
            rows={5}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="e.g. I bought this for my kitchen and it works great, the build quality is solid and it has lasted me over a year without any issues at all."
            className="w-full bg-[#212B31] border border-[#2E3A41] rounded-lg p-4 text-sm text-[#E9EDEE] placeholder-[#8DA0A8]/50 focus:outline-none focus:border-[#5FD3A0] transition-colors"
          ></textarea>
        </div>

        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <span className="text-xs font-mono text-[#8DA0A8]">
            {wordCount} words {wordCount < 3 && '(minimum 3 required)'}
          </span>

          <button
            onClick={handleAnalyze}
            disabled={isButtonDisabled}
            className={`inline-flex items-center justify-center gap-2 px-6 py-3 rounded-lg font-mono font-bold text-sm transition-all shadow-md ${
              isButtonDisabled
                ? 'bg-[#2E3A41] text-[#8DA0A8] cursor-not-allowed'
                : 'bg-[#E9EDEE] text-[#12181C] hover:bg-[#5FD3A0] cursor-pointer'
            }`}
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin text-[#12181C]" />
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

      {/* Error alert */}
      {error && (
        <div className="bg-[#E64980]/15 border border-[#E64980] rounded-xl p-4 flex items-center gap-3 text-sm text-[#E64980] font-mono">
          <AlertTriangle className="w-5 h-5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Prediction Output Results */}
      {result && (
        <div className="space-y-6 animate-fadeIn">
          {/* Verdict Box */}
          <div
            className={`rounded-xl p-6 border transition-all ${
              result.label === 'real'
                ? 'bg-[#5FD3A0]/10 border-[#5FD3A0]/40'
                : 'bg-[#E64980]/10 border-[#E64980]/40'
            }`}
          >
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
              <div>
                <div
                  className={`font-mono text-xl sm:text-2xl font-bold flex items-center gap-2 ${
                    result.label === 'real' ? 'text-[#5FD3A0]' : 'text-[#E64980]'
                  }`}
                >
                  {result.label === 'real' ? (
                    <>
                      <CheckCircle2 className="w-7 h-7" />
                      ✓ LIKELY GENUINE
                    </>
                  ) : (
                    <>
                      <AlertTriangle className="w-7 h-7" />
                      ⚑ LIKELY AI-GENERATED
                    </>
                  )}
                </div>
                <div className="text-xs sm:text-sm text-[#8DA0A8] font-mono mt-1">
                  Model confidence: <strong className="text-[#E9EDEE]">{(result.confidence * 100).toFixed(1)}%</strong>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <span className="text-xs font-mono text-[#8DA0A8] bg-[#12181C] border border-[#2E3A41] px-3 py-1.5 rounded-lg">
                  Device: {result.device}
                </span>
              </div>
            </div>
          </div>

          {/* Probability Meters */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="bg-[#1B2328] border border-[#2E3A41] rounded-xl p-5">
              <div className="flex justify-between items-center mb-2 font-mono">
                <span className="text-xs text-[#8DA0A8]">Real Probability</span>
                <span className="text-lg font-bold text-[#5FD3A0]">{(result.prob_real * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-[#212B31] h-3 rounded-full overflow-hidden">
                <div
                  className="bg-[#5FD3A0] h-full transition-all duration-500"
                  style={{ width: `${result.prob_real * 100}%` }}
                ></div>
              </div>
            </div>

            <div className="bg-[#1B2328] border border-[#2E3A41] rounded-xl p-5">
              <div className="flex justify-between items-center mb-2 font-mono">
                <span className="text-xs text-[#8DA0A8]">Fake Probability</span>
                <span className="text-lg font-bold text-[#E64980]">{(result.prob_fake * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-[#212B31] h-3 rounded-full overflow-hidden">
                <div
                  className="bg-[#E64980] h-full transition-all duration-500"
                  style={{ width: `${result.prob_fake * 100}%` }}
                ></div>
              </div>
            </div>
          </div>

          {/* SHAP / Feature Explanation Chart */}
          <div className="bg-[#1B2328] border border-[#2E3A41] rounded-xl p-6 space-y-6">
            <div>
              <h3 className="text-lg font-bold text-[#E9EDEE] flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-[#F4C95D]" />
                🔬 Why did the model make this prediction?
              </h3>
              <p className="text-xs sm:text-sm text-[#8DA0A8] mt-1">
                Which words/features pushed this review toward FAKE vs. REAL
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
                    <CartesianGrid strokeDasharray="3 3" stroke="#2E3A41" />
                    <XAxis type="number" stroke="#8DA0A8" tick={{ fill: '#8DA0A8', fontSize: 11 }} />
                    <YAxis dataKey="display_token" type="category" stroke="#8DA0A8" tick={{ fill: '#E9EDEE', fontSize: 12 }} width={80} />
                    <ReferenceLine x={0} stroke="#8DA0A8" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#212B31', borderColor: '#2E3A41', color: '#E9EDEE', borderRadius: '8px' }}
                      itemStyle={{ color: '#E9EDEE' }}
                      labelStyle={{ color: '#8DA0A8', fontWeight: 600 }}
                      cursor={{ fill: 'rgba(255, 255, 255, 0.05)' }}
                      formatter={(val) => [val, 'Contribution']}
                    />
                    <Bar dataKey="shap_value" radius={[2, 2, 2, 2]}>
                      {result.contributions.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.shap_value > 0 ? '#E64980' : '#5FD3A0'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="bg-[#212B31] border border-[#2E3A41] rounded-lg p-6 text-center text-xs text-[#8DA0A8] font-mono">
                No strong word/feature contributions extracted for this text.
              </div>
            )}

            {/* Chart Legend Box */}
            <div className="bg-[#212B31] border border-[#2E3A41] rounded-xl p-5 text-xs text-[#8DA0A8] space-y-2">
              <div className="font-mono text-sm font-bold text-[#E9EDEE]">🔬 How to read this chart</div>
              <p>
                <span className="text-[#E64980] font-bold">Pink bars</span> push the prediction toward <b>FAKE</b>.{' '}
                <span className="text-[#5FD3A0] font-bold">Green bars</span> push it toward <b>REAL</b>.
              </p>
              <p>
                This coloring is fixed regardless of what the model predicted for this review. Longer bars had a bigger effect on this specific prediction.
              </p>
            </div>

            {/* Top Influential Table */}
            {result.top_influential && result.top_influential.length > 0 && (
              <div className="space-y-3 pt-4">
                <h4 className="text-sm font-bold text-[#E9EDEE]">🔍 Most influential words / features</h4>
                <div className="overflow-x-auto border border-[#2E3A41] rounded-lg">
                  <table className="w-full text-left text-xs font-mono">
                    <thead className="bg-[#212B31] text-[#8DA0A8] border-b border-[#2E3A41]">
                      <tr>
                        <th className="p-3">Word / Feature</th>
                        <th className="p-3">Contribution</th>
                        <th className="p-3">Pushes toward</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#2E3A41] text-[#E9EDEE]">
                      {result.top_influential.map((row, idx) => (
                        <tr key={idx} className="hover:bg-[#212B31]/50">
                          <td className="p-3 font-semibold">{row.token}</td>
                          <td className="p-3">{row.contribution}</td>
                          <td className={`p-3 font-bold ${row.pushes_toward.includes('FAKE') ? 'text-[#E64980]' : 'text-[#5FD3A0]'}`}>
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
