import React, { useState } from 'react';
import {
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend
} from 'recharts';
import { Link2, Loader2, ShieldCheck, AlertTriangle, ChevronDown, ChevronUp, Sparkles, CheckCircle2, Construction } from 'lucide-react';
import { formatPercent } from '../utils/format';

export default function UrlScraperTab() {
  const [url, setUrl] = useState('');
  const [modelChoice, setModelChoice] = useState('DistilBERT (Transformer)');
  const [maxReviews, setMaxReviews] = useState(10);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [expandedIndex, setExpandedIndex] = useState(null);

  const sampleUrls = [
    { label: 'Sample Amazon Link', url: 'https://www.amazon.com/dp/B08N5WRWNW' },
    { label: 'Sample Product Page', url: 'https://www.amazon.com/dp/B09B8V1LZ3' },
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
      {/* Under Construction Banner */}
      <div className="bg-amber-50/80 border border-amber-200 rounded-xl p-4 sm:p-5 flex items-start sm:items-center gap-3.5 text-amber-900 shadow-2xs">
        <div className="p-2.5 bg-amber-100/90 rounded-lg text-amber-700 shrink-0 mt-0.5 sm:mt-0">
          <Construction className="w-5 h-5" />
        </div>
        <div className="flex-1 space-y-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-bold text-sm text-amber-900 tracking-tight">Under Construction</span>
            <span className="text-[10px] font-mono uppercase bg-amber-200/80 border border-amber-300 text-amber-800 px-2 py-0.5 rounded font-semibold tracking-wider">
              Prototype / Experimental Feature
            </span>
          </div>
          <p className="text-xs text-amber-800 leading-relaxed">
            The automated URL scraper is currently under construction and active development. E-commerce sites often utilize anti-bot protection and CAPTCHAs, which may cause live scraping requests to be blocked.
          </p>
        </div>
      </div>

      {/* Header */}
      <div>
        <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-indigo-50 border border-indigo-100 text-indigo-700 text-xs font-semibold mb-2">
          <Link2 className="w-3.5 h-3.5 text-indigo-600" />
          E-COMMERCE URL AUTO-SCRAPER & TRUST SCORE
        </div>
        <h2 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
          Scan product URLs in real time
        </h2>
        <p className="text-sm text-slate-500 mt-1">
          Paste any e-commerce product URL to automatically extract reviews, batch-classify authenticity, and compute a Product Trust Score.
        </p>
      </div>

      {/* Input Form */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-5 shadow-2xs">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="w-full sm:w-72">
            <label className="block text-xs font-mono text-slate-500 mb-1">Select Classification Model</label>
            <select
              value={modelChoice}
              onChange={(e) => setModelChoice(e.target.value)}
              className="w-full bg-white border border-slate-300 rounded-lg px-3 py-2 text-xs font-semibold text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            >
              <option value="DistilBERT (Transformer)">DistilBERT (Transformer)</option>
              <option value="Fusion - Logistic Regression">Fusion - Logistic Regression</option>
              <option value="Fusion - Linear SVM">Fusion - Linear SVM</option>
            </select>
          </div>

          <div className="w-full sm:w-40">
            <label className="block text-xs font-mono text-slate-500 mb-1">Max Reviews</label>
            <select
              value={maxReviews}
              onChange={(e) => setMaxReviews(e.target.value)}
              className="w-full bg-white border border-slate-300 rounded-lg px-3 py-2 text-xs font-semibold text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            >
              <option value={5}>5 Reviews</option>
              <option value={10}>10 Reviews</option>
              <option value={15}>15 Reviews</option>
            </select>
          </div>

          {/* Quick Presets */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500 font-mono">Sample:</span>
            {sampleUrls.map((s, idx) => (
              <button
                key={idx}
                onClick={() => setUrl(s.url)}
                className="text-xs font-mono bg-slate-50 border border-slate-300 hover:bg-slate-100 text-slate-700 px-2.5 py-1 rounded transition-all cursor-pointer"
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="block text-xs font-mono text-slate-500 mb-1">E-Commerce Product URL</label>
          <div className="relative">
            <input
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="e.g. https://www.amazon.com/dp/B08N5WRWNW"
              className="w-full bg-white border border-slate-300 rounded-xl pl-10 pr-4 py-3 text-sm text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 shadow-2xs"
            />
            <Link2 className="w-4 h-4 text-slate-400 absolute left-3.5 top-3.5" />
          </div>
        </div>

        <button
          onClick={handleScan}
          disabled={!url.trim() || loading}
          className={`w-full sm:w-auto inline-flex items-center justify-center gap-2 px-6 py-3 rounded-lg font-semibold text-sm transition-all shadow-sm ${
            !url.trim() || loading
              ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
              : 'bg-indigo-600 hover:bg-indigo-700 text-white cursor-pointer'
          }`}
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin text-white" />
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
        <div className="bg-rose-50 border border-rose-200 rounded-xl p-4 flex items-center gap-3 text-sm text-rose-700 font-mono">
          <AlertTriangle className="w-5 h-5 shrink-0 text-rose-600" />
          <span>{error}</span>
        </div>
      )}

      {/* Scraped Results View */}
      {result && (
        <div className="space-y-8 animate-fadeIn">
          {/* Header Product Info & Trust Score Card */}
          <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-2xs relative overflow-hidden">
            <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
              <div className="space-y-2 max-w-2xl">
                <span className="text-xs font-mono text-slate-500 uppercase tracking-wider">Scraped Product Target</span>
                <h3 className="text-xl font-extrabold text-slate-900 line-clamp-2">{result.product_title}</h3>
                <a
                  href={result.url}
                  target="_blank"
                  rel="noreferrer"
                  className="text-xs text-indigo-600 hover:underline font-mono truncate block"
                >
                  {result.url}
                </a>
              </div>

              {/* Trust Score Gauge Card */}
              <div className="bg-slate-50 border border-slate-200 rounded-xl p-5 flex items-center gap-6 shrink-0">
                <div className="text-center font-mono">
                  <div
                    className={`text-4xl font-extrabold ${
                      result.trust_score >= 70
                        ? 'text-emerald-600'
                        : result.trust_score >= 40
                        ? 'text-amber-600'
                        : 'text-rose-600'
                    }`}
                  >
                    {result.trust_score}%
                  </div>
                  <div className="text-[11px] text-slate-500 mt-1 font-semibold uppercase">Product Trust Score</div>
                </div>

                <div className="border-l border-slate-200 pl-5 text-xs font-mono space-y-1.5">
                  <div className="flex items-center gap-2 text-emerald-700 font-semibold">
                    <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                    <span>{result.real_count} Genuine Reviews</span>
                  </div>
                  <div className="flex items-center gap-2 text-rose-700 font-semibold">
                    <AlertTriangle className="w-4 h-4 text-rose-600" />
                    <span>{result.fake_count} Flagged AI Reviews</span>
                  </div>
                  <div className="text-slate-500 text-[11px] pt-0.5">
                    Total Evaluated: {result.total_reviews} reviews
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Breakdown Charts */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Pie Chart */}
            <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-2xs flex flex-col items-center">
              <h4 className="text-base font-bold text-slate-900 w-full text-left mb-4">Product Review Authenticity</h4>
              <div className="w-full h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={[
                        { name: 'Genuine Real', value: result.real_count, color: '#059669' },
                        { name: 'AI-Generated Fake', value: result.fake_count, color: '#E11D48' },
                      ]}
                      cx="50%"
                      cy="50%"
                      innerRadius={50}
                      outerRadius={80}
                      paddingAngle={4}
                      dataKey="value"
                    >
                      <Cell fill="#059669" />
                      <Cell fill="#E11D48" />
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#FFFFFF',
                        borderColor: '#E2E8F0',
                        color: '#0F172A',
                        borderRadius: '8px',
                        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                        fontSize: '12px',
                      }}
                    />
                    <Legend formatter={(val) => <span className="text-slate-600 text-xs font-medium">{val}</span>} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Quick Summary Card */}
            <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-2xs flex flex-col justify-between space-y-4">
              <h4 className="text-base font-bold text-slate-900">Automated Risk Assessment</h4>
              <p className="text-sm text-slate-600 leading-relaxed">
                {result.trust_score >= 70
                  ? '✅ High Trust: The majority of reviews extracted on this product page exhibit natural human language signatures, varied sentence structure, and authentic personal experiences.'
                  : result.trust_score >= 40
                  ? '⚠️ Moderate Risk: A notable portion of reviews exhibit repetitive phrasing and synthetic structures characteristic of AI text generators.'
                  : '🚨 High AI Risk: A significant majority of reviews on this product page display strong synthetic AI-generation patterns.'}
              </p>

              <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-xs text-slate-600 font-mono">
                Classification Model Used: <strong>{modelChoice}</strong>
              </div>
            </div>
          </div>

          {/* Individual Analyzed Reviews List */}
          <div className="space-y-4">
            <h4 className="text-base font-bold text-slate-900">Analyzed Product Reviews ({result.reviews.length})</h4>

            {result.reviews.map((item, idx) => {
              const isReal = item.label === 'real';
              const isExpanded = expandedIndex === idx;

              return (
                <div
                  key={idx}
                  className="bg-white border border-slate-200 rounded-xl p-5 space-y-3.5 transition-all shadow-2xs hover:shadow-md"
                >
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                    <div className="flex items-center gap-3">
                      <span className="text-amber-500 font-bold text-xs font-mono">
                        {'★'.repeat(item.rating)}{'☆'.repeat(5 - item.rating)}
                      </span>
                      <h5 className="font-bold text-slate-900 text-sm">{item.title || `Review #${idx + 1}`}</h5>
                    </div>

                    <span
                      className={`inline-flex items-center gap-1.5 font-mono text-xs font-bold px-2.5 py-1 rounded border self-start sm:self-auto ${
                        isReal
                          ? 'bg-emerald-50 border-emerald-200 text-emerald-700'
                          : 'bg-rose-50 border-rose-200 text-rose-700'
                      }`}
                    >
                      {isReal ? '✓ GENUINE' : '⚑ AI-GENERATED'} ({formatPercent(item.confidence)})
                    </span>
                  </div>

                  <p className="text-sm text-slate-800 leading-relaxed bg-slate-50 border-l-4 border-slate-300 p-3 rounded-r-lg">
                    &ldquo;{item.text}&rdquo;
                  </p>

                  <div className="flex items-center justify-between pt-2 border-t border-slate-100">
                    <div className="flex items-center gap-4 text-xs font-mono text-slate-500">
                      <span>Real: {formatPercent(item.prob_real)}</span>
                      <span>Fake: {formatPercent(item.prob_fake)}</span>
                    </div>

                    {item.top_influential && item.top_influential.length > 0 && (
                      <button
                        onClick={() => setExpandedIndex(isExpanded ? null : idx)}
                        className="inline-flex items-center gap-1 text-xs font-mono text-indigo-600 hover:text-indigo-800 font-semibold cursor-pointer"
                      >
                        <Sparkles className="w-3.5 h-3.5 text-amber-500" />
                        {isExpanded ? 'Hide Keywords' : 'Top Keywords'}
                        {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                      </button>
                    )}
                  </div>

                  {/* Expandable Token Table */}
                  {isExpanded && item.top_influential && (
                    <div className="pt-2 animate-fadeIn">
                      <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 space-y-2">
                        <div className="text-xs font-bold text-slate-900 font-mono">Top Influential Keywords:</div>
                        <div className="flex flex-wrap gap-2">
                          {item.top_influential.map((tokenObj, tIdx) => (
                            <span
                              key={tIdx}
                              className={`text-xs font-mono px-2 py-0.5 rounded border ${
                                tokenObj.pushes_toward.includes('FAKE')
                                  ? 'bg-rose-50 border-rose-200 text-rose-700'
                                  : 'bg-emerald-50 border-emerald-200 text-emerald-700'
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
