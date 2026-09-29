import React from 'react';
import { ArrowRight, CheckCircle2, ShieldCheck, Zap, Layers, Sparkles, Database, Award } from 'lucide-react';

export default function OverviewTab({ stats, setActiveTab }) {
  const defaultStats = [
    { value: '40,405', label: 'Reviews Analyzed', icon: Database, color: 'text-indigo-600', bg: 'bg-indigo-50 border-indigo-100' },
    { value: '10', label: 'Product Categories', icon: Layers, color: 'text-blue-600', bg: 'bg-blue-50 border-blue-100' },
    { value: '4', label: 'Modeling Approaches', icon: Zap, color: 'text-amber-600', bg: 'bg-amber-50 border-amber-100' },
    { value: '98.05%', label: 'Best Model Accuracy', icon: Award, color: 'text-emerald-600', bg: 'bg-emerald-50 border-emerald-100' },
  ];

  const displayStats = stats
    ? stats.map((s, idx) => ({
        ...s,
        icon: defaultStats[idx]?.icon || ShieldCheck,
        color: defaultStats[idx]?.color || 'text-indigo-600',
        bg: defaultStats[idx]?.bg || 'bg-indigo-50 border-indigo-100',
      }))
    : defaultStats;

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Hero section */}
      <div className="bg-white border border-slate-200 rounded-2xl p-6 sm:p-8 relative overflow-hidden shadow-xs">
        <div className="absolute top-0 right-0 w-96 h-96 bg-indigo-500/5 rounded-full blur-3xl pointer-events-none -mr-20 -mt-20"></div>
        <div className="max-w-3xl space-y-4 relative z-10">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-50 border border-indigo-100 text-indigo-700 text-xs font-semibold">
            <Sparkles className="w-3.5 h-3.5 text-indigo-600" />
            AI-POWERED DECEPTION DETECTION PIPELINE
          </div>
          <h2 className="text-2xl sm:text-4xl font-extrabold text-slate-900 tracking-tight leading-tight">
            Detecting deception in product reviews
          </h2>
          <p className="text-slate-600 text-base sm:text-lg leading-relaxed font-normal">
            A full end-to-end investigation trail from raw Amazon review data to a fine-tuned DistilBERT transformer built to accurately separate authentic customer feedback from AI-generated fake reviews.
          </p>
          <div className="pt-2 flex flex-wrap gap-3">
            <button
              onClick={() => setActiveTab('LiveInterrogation')}
              className="inline-flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white px-5 py-2.5 rounded-lg font-semibold text-sm transition-colors shadow-sm cursor-pointer"
            >
              Try Live Interrogation
              <ArrowRight className="w-4 h-4" />
            </button>
            <button
              onClick={() => setActiveTab('Dataset')}
              className="inline-flex items-center gap-2 bg-white border border-slate-300 hover:bg-slate-50 text-slate-700 px-5 py-2.5 rounded-lg font-medium text-sm transition-colors shadow-2xs cursor-pointer"
            >
              Explore Dataset & EDA
            </button>
          </div>
        </div>
      </div>

      {/* KPI Metrics Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
        {displayStats.map((stat, idx) => {
          const Icon = stat.icon;
          return (
            <div
              key={idx}
              className="bg-white border border-slate-200 rounded-xl p-5 sm:p-6 transition-all hover:shadow-md hover:border-slate-300 space-y-3 shadow-2xs"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs sm:text-sm font-semibold text-slate-500">
                  {stat.label}
                </span>
                <div className={`p-2 rounded-lg border ${stat.bg}`}>
                  <Icon className={`w-4 h-4 ${stat.color}`} />
                </div>
              </div>
              <div className="font-mono text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight">
                {stat.value}
              </div>
            </div>
          );
        })}
      </div>

      {/* Summary Highlight Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-3.5 shadow-2xs hover:shadow-md transition-shadow">
          <div className="w-10 h-10 rounded-lg bg-emerald-50 border border-emerald-100 flex items-center justify-center text-emerald-600">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <h3 className="text-base font-bold text-slate-900">Transformer Fine-Tuning</h3>
          <p className="text-sm text-slate-600 leading-relaxed">
            DistilBERT fine-tuning delivers an outstanding <strong className="text-slate-900">98.05% accuracy</strong>, effectively identifying GPT-2 synthetic generation artifacts across 10 e-commerce categories.
          </p>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-3.5 shadow-2xs hover:shadow-md transition-shadow">
          <div className="w-10 h-10 rounded-lg bg-amber-50 border border-amber-100 flex items-center justify-center text-amber-600">
            <Layers className="w-5 h-5" />
          </div>
          <h3 className="text-base font-bold text-slate-900">Hybrid Fusion Approach</h3>
          <p className="text-sm text-slate-600 leading-relaxed">
            Combines 5,000 TF-IDF n-grams with 16 hand-engineered linguistic and POS signals (<strong className="text-slate-900">5,016 total dimensions</strong>) achieving 90.89% baseline accuracy.
          </p>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-3.5 shadow-2xs hover:shadow-md transition-shadow">
          <div className="w-10 h-10 rounded-lg bg-indigo-50 border border-indigo-100 flex items-center justify-center text-indigo-600">
            <Zap className="w-5 h-5" />
          </div>
          <h3 className="text-base font-bold text-slate-900">XAI & Error Analysis</h3>
          <p className="text-sm text-slate-600 leading-relaxed">
            Integrated SHAP token contribution visualizer explains exact model reasoning for every review, coupled with error analysis on false positive and negative edge cases.
          </p>
        </div>
      </div>

      {/* Info Banner */}
      <div className="bg-slate-100 border border-slate-200 rounded-xl p-4 sm:p-5 flex flex-col sm:flex-row items-center justify-between text-xs sm:text-sm text-slate-600 gap-2">
        <span className="font-medium">
          Capstone Project in Data Science II (DS4105) - Sabaragamuwa University of Sri Lanka
        </span>
        <span className="font-mono text-emerald-700 bg-emerald-50 px-2.5 py-1 rounded border border-emerald-200 font-semibold">
          Verified Research Pipeline
        </span>
      </div>
    </div>
  );
}
