import React, { useState } from 'react';
import {
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend,
  BarChart, Bar, XAxis, YAxis, CartesianGrid
} from 'recharts';
import { Link2, Loader2, ShieldCheck, AlertTriangle, ChevronDown, ChevronUp, Sparkles, CheckCircle2 } from 'lucide-react';

export default function UrlScraperTab() {
  const [url, setUrl] = useState('');
  const [modelChoice, setModelChoice] = useState('DistilBERT (Transformer)');
  const [maxReviews, setMaxReviews] = useState(10);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [expandedIndex, setExpandedIndex] = useState(null);

  const sampleUrls = [
    { label: "Sample Amazon Link", url: "https://www.amazon.com/dp/B08N5WRWNW" },
    { label: "Sample Product Page", url: "https://www.amazon.com/dp/B09B8V1LZ3" },
  ];

  const handleScan = async () => {
    if (!url.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('/api/scrape-and-predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          url: url.trim(),
          model_choice: modelChoice,
          max_reviews: parseInt(maxReviews),
        }),
      });

      let data;
      try {
        data = await response.json();
      } catch (parseErr) {
        throw new Error(`Server returned HTTP ${response.status}. Check backend logs.`);
      }

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to scrape and analyze URL.');
      }

      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div>
        <div className="inline-flex items-center gap-2 font-mono text-xs text-[#8DA0A8] border border-[#2E3A41] px-2.5 py-1 rounded bg-[#1B2328] mb-2">
          <span className="w-2 h-2 rounded-full bg-[#5FD3A0]"></span>
          E-COMMERCE URL AUTO-SCRAPER & TRUST SCORE
        </div>
        <h2 className="text-2xl sm:text-3xl font-extrabold text-[#E9EDEE]">
          Scan product URLs in real time
        </h2>
        <p className="text-sm text-[#8DA0A8] mt-1">
          Paste any e-commerce product URL to automatically extract, batch-classify reviews, and compute an overall Product Trust Score.
        </p>
      </div>

      {/* Input Form */}
      <div className="bg-[#1B2328] border border-[#2E3A41] rounded-xl p-6 space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="w-full sm:w-72">
            <label className="block text-xs font-mono text-[#8DA0A8] mb-1">Select Classification Model</label>
            <select
              value={modelChoice}
              onChange={(e) => setModelChoice(e.target.value)}
              className="w-full bg-[#212B31] border border-[#2E3A41] rounded-lg px-3 py-2 text-xs font-semibold text-[#E9EDEE] focus:outline-none focus:border-[#5FD3A0]"
            >
              <option value="DistilBERT (Transformer)">DistilBERT (Transformer)</option>
              <option value="Fusion — Logistic Regression">Fusion — Logistic Regression</option>
              <option value="Fusion — Linear SVM">Fusion — Linear SVM</option>
            </select>
          </div>

          <div className="w-full sm:w-40">
            <label className="block text-xs font-mono text-[#8DA0A8] mb-1">Max Reviews</label>
            <select
              value={maxReviews}
              onChange={(e) => setMaxReviews(e.target.value)}
              className="w-full bg-[#212B31] border border-[#2E3A41] rounded-lg px-3 py-2 text-xs font-semibold text-[#E9EDEE] focus:outline-none focus:border-[#5FD3A0]"
            >
              <option value={5}>5 Reviews</option>
              <option value={10}>10 Reviews</option>
              <option value={15}>15 Reviews</option>
            </select>
          </div>

          {/* Quick Presets */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-[#8DA0A8] font-mono">Sample:</span>
            {sampleUrls.map((s, idx) => (
              <button
                key={idx}
                onClick={() => setUrl(s.url)}
                className="text-xs font-mono bg-[#212B31] border border-[#2E3A41] hover:border-[#8DA0A8] text-[#E9EDEE] px-2.5 py-1 rounded transition-all"
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="block text-xs font-mono text-[#8DA0A8] mb-1">E-Commerce Product URL</label>
          <div className="relative">
            <input
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="e.g. https://www.amazon.com/dp/B08N5WRWNW"
              className="w-full bg-[#212B31] border border-[#2E3A41] rounded-lg pl-10 pr-4 py-3 text-sm text-[#E9EDEE] placeholder-[#8DA0A8]/50 focus:outline-none focus:border-[#5FD3A0]"
            />
            <Link2 className="w-4 h-4 text-[#8DA0A8] absolute left-3.5 top-3.5" />
          </div>
        </div>

        <button
          onClick={handleScan}
          disabled={!url.trim() || loading}
          className={`w-full sm:w-auto inline-flex items-center justify-center gap-2 px-6 py-3 rounded-lg font-mono font-bold text-sm transition-all shadow-md ${
            !url.trim() || loading
              ? 'bg-[#2E3A41] text-[#8DA0A8] cursor-not-allowed'
              : 'bg-[#E9EDEE] text-[#12181C] hover:bg-[#5FD3A0] cursor-pointer'
          }`}
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin text-[#12181C]" />
              Scraping & Analyzing Reviews...
            </>
          ) : (
            <>
              <ShieldCheck className="w-4 h-4" />
              Scan Product Reviews
            </>
          )}
        </button>
      </div>

      {/* Error alert */}
      {error && (
        <div className="bg-[#E64980]/15 border border-[#E64980] rounded-xl p-4 flex items-center gap-3 text-sm text-[#E64980] font-mono">
          <AlertTriangle className="w-5 h-5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Scraped Results View */}
      {result && (
        <div className="space-y-8 animate-fadeIn">
          {/* Header Product Info & Trust Score Card */}
          <div className="bg-[#1B2328] border border-[#2E3A41] rounded-xl p-6 relative overflow-hidden">
            <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
              <div className="space-y-2 max-w-2xl">
                <span className="text-xs font-mono text-[#8DA0A8] uppercase tracking-wider">Scraped Product Target</span>
                <h3 className="text-xl font-extrabold text-[#E9EDEE] line-clamp-2">{result.product_title}</h3>
                <a
                  href={result.url}
                  target="_blank"
                  rel="noreferrer"
                  className="text-xs text-[#5FD3A0] hover:underline font-mono truncate block"
                >
                  {result.url}
                </a>
              </div>

              {/* Trust Score Gauge Card */}
              <div className="bg-[#212B31] border border-[#2E3A41] rounded-xl p-5 flex items-center gap-6 shrink-0">
                <div className="text-center font-mono">
                  <div
                    className={`text-4xl font-extrabold ${
                      result.trust_score >= 70
                        ? 'text-[#5FD3A0]'
                        : result.trust_score >= 40
                        ? 'text-[#F4C95D]'
                        : 'text-[#E64980]'
                    }`}
                  >
                    {result.trust_score}%
                  </div>
                  <div className="text-[11px] text-[#8DA0A8] mt-1 font-semibold uppercase">Product Trust Score</div>
                </div>

                <div className="border-l border-[#2E3A41] pl-5 text-xs font-mono space-y-1">
                  <div className="flex items-center gap-2 text-[#5FD3A0]">
                    <CheckCircle2 className="w-4 h-4" />
                    <span>{result.real_count} Genuine Reviews</span>
                  </div>
                  <div className="flex items-center gap-2 text-[#E64980]">
                    <AlertTriangle className="w-4 h-4" />
                    <span>{result.fake_count} Flagged AI Reviews</span>
                  </div>
                  <div className="text-[#8DA0A8] text-[11px] pt-1">
                    Total Evaluated: {result.total_reviews} reviews
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Breakdown Charts */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Pie Chart */}
            <div className="bg-[#1B2328] border border-[#2E3A41] rounded-xl p-5 flex flex-col items-center">
              <h4 className="text-sm font-bold text-[#E9EDEE] w-full text-left mb-4">Product Review Authenticity</h4>
              <div className="w-full h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={[
                        { name: 'Genuine Real', value: result.real_count, color: '#5FD3A0' },
                        { name: 'AI-Generated Fake', value: result.fake_count, color: '#E64980' },
                      ]}
                      cx="50%"
                      cy="50%"
                      innerRadius={50}
                      outerRadius={80}
                      paddingAngle={4}
                      dataKey="value"
                    >
                      <Cell fill="#5FD3A0" />
                      <Cell fill="#E64980" />
                    </Pie>
                    <Tooltip
                      contentStyle={{ backgroundColor: '#212B31', borderColor: '#2E3A41', color: '#E9EDEE', borderRadius: '8px' }}
                      itemStyle={{ color: '#E9EDEE' }}
                      labelStyle={{ color: '#8DA0A8', fontWeight: 600 }}
                      cursor={{ fill: 'rgba(255, 255, 255, 0.05)' }}
                    />
                    <Legend formatter={(val) => <span className="text-[#8DA0A8] text-xs font-mono">{val}</span>} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Quick Summary Card */}
            <div className="bg-[#1B2328] border border-[#2E3A41] rounded-xl p-6 flex flex-col justify-between space-y-4">
              <h4 className="text-sm font-bold text-[#E9EDEE]">Automated Risk Assessment</h4>
              <p className="text-xs text-[#8DA0A8] leading-relaxed">
                {result.trust_score >= 70
                  ? "✅ High Trust: The majority of reviews extracted on this product page exhibit natural human language signatures, varied sentence structure, and personal anecdotes."
                  : result.trust_score >= 40
                  ? "⚠️ Moderate Risk: A notable portion of reviews exhibit repetitive phrasing and synthetic structures characteristic of AI text generators."
                  : "🚨 High AI Risk: A significant majority of reviews on this product page display strong synthetic AI-generation patterns."}
              </p>

              <div className="bg-[#212B31] border border-[#2E3A41] rounded-lg p-3 text-xs text-[#8DA0A8] font-mono">
                Model: <strong>{modelChoice}</strong>
              </div>
            </div>
          </div>

          {/* Individual Analyzed Reviews List */}
          <div className="space-y-4">
            <h4 className="text-base font-bold text-[#E9EDEE]">Analyzed Product Reviews ({result.reviews.length})</h4>

            {result.reviews.map((item, idx) => {
              const isReal = item.label === 'real';
              const isExpanded = expandedIndex === idx;

              return (
                <div
                  key={idx}
                  className="bg-[#1B2328] border border-[#2E3A41] rounded-xl p-5 space-y-4 transition-all hover:border-[#8DA0A8]/40"
                >
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                    <div className="flex items-center gap-3">
                      <span className="text-yellow-400 font-bold text-xs font-mono">
                        {'★'.repeat(item.rating)}{'☆'.repeat(5 - item.rating)}
                      </span>
                      <h5 className="font-bold text-[#E9EDEE] text-sm">{item.title || `Review #${idx + 1}`}</h5>
                    </div>

                    <span
                      className={`inline-flex items-center gap-1.5 font-mono text-xs font-bold px-2.5 py-1 rounded border self-start sm:self-auto ${
                        isReal
                          ? 'bg-[#5FD3A0]/10 border-[#5FD3A0]/40 text-[#5FD3A0]'
                          : 'bg-[#E64980]/10 border-[#E64980]/40 text-[#E64980]'
                      }`}
                    >
                      {isReal ? '✓ GENUINE' : '⚑ AI-GENERATED'} ({(item.confidence * 100).toFixed(1)}%)
                    </span>
                  </div>

                  <p className="text-sm text-[#E9EDEE] leading-relaxed pl-3 border-l-2 border-[#2E3A41]">
                    &ldquo;{item.text}&rdquo;
                  </p>

                  <div className="flex items-center justify-between pt-2 border-t border-[#2E3A41]/60">
                    <div className="flex items-center gap-4 text-xs font-mono text-[#8DA0A8]">
                      <span>Real: {(item.prob_real * 100).toFixed(1)}%</span>
                      <span>Fake: {(item.prob_fake * 100).toFixed(1)}%</span>
                    </div>

                    {item.top_influential && item.top_influential.length > 0 && (
                      <button
                        onClick={() => setExpandedIndex(isExpanded ? null : idx)}
                        className="inline-flex items-center gap-1 text-xs font-mono text-[#8DA0A8] hover:text-[#E9EDEE]"
                      >
                        <Sparkles className="w-3.5 h-3.5 text-[#F4C95D]" />
                        {isExpanded ? 'Hide Keywords' : 'Top Keywords'}
                        {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                      </button>
                    )}
                  </div>

                  {/* Expandable Token Table */}
                  {isExpanded && item.top_influential && (
                    <div className="pt-3 animate-fadeIn">
                      <div className="bg-[#212B31] border border-[#2E3A41] rounded-lg p-3 space-y-2">
                        <div className="text-xs font-bold text-[#E9EDEE] font-mono">Top Influential Keywords:</div>
                        <div className="flex flex-wrap gap-2">
                          {item.top_influential.map((tokenObj, tIdx) => (
                            <span
                              key={tIdx}
                              className={`text-xs font-mono px-2 py-0.5 rounded border ${
                                tokenObj.pushes_toward.includes('FAKE')
                                  ? 'bg-[#E64980]/10 border-[#E64980]/40 text-[#E64980]'
                                  : 'bg-[#5FD3A0]/10 border-[#5FD3A0]/40 text-[#5FD3A0]'
                              }`}
                            >
                              {tokenObj.token} ({tokenObj.contribution > 0 ? `+${tokenObj.contribution}` : tokenObj.contribution})
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
