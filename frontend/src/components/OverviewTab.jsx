import React from 'react';
import { ArrowRight, CheckCircle2, ShieldCheck, Zap, Layers } from 'lucide-react';

export default function OverviewTab({ stats, setActiveTab }) {
  const defaultStats = [
    { value: "40,405", label: "reviews analyzed" },
    { value: "10", label: "product categories" },
    { value: "4", label: "modeling approaches tested" },
    { value: "98.3%", label: "best model accuracy" },
  ];

  const displayStats = stats || defaultStats;

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Hero section */}
      <div className="bg-[#1B2328] border border-[#2E3A41] rounded-xl p-6 sm:p-8 relative overflow-hidden">
        <div className="absolute -right-12 -bottom-12 w-64 h-64 bg-[#5FD3A0]/5 rounded-full blur-3xl pointer-events-none"></div>
        <div className="max-w-3xl space-y-4">
          <h2 className="text-2xl sm:text-4xl font-extrabold text-[#E9EDEE] tracking-tight">
            Detecting deception in product reviews
          </h2>
          <p className="text-[#8DA0A8] text-base sm:text-lg leading-relaxed">
            A full investigation trail - from raw review data to a fine-tuned transformer - built to tell genuine customer reviews apart from AI-generated ones.
          </p>
          <div className="pt-2 flex flex-wrap gap-4">
            <button
              onClick={() => setActiveTab('LiveInterrogation')}
              className="inline-flex items-center gap-2 bg-[#E9EDEE] text-[#12181C] px-5 py-2.5 rounded-lg font-mono font-bold text-sm hover:bg-[#5FD3A0] transition-colors shadow-lg"
            >
              🔎 Try Live Interrogation
              <ArrowRight className="w-4 h-4" />
            </button>
            <button
              onClick={() => setActiveTab('Dataset')}
              className="inline-flex items-center gap-2 bg-[#212B31] border border-[#2E3A41] text-[#E9EDEE] px-5 py-2.5 rounded-lg font-mono text-sm hover:bg-[#2E3A41] transition-colors"
            >
              Explore Dataset & EDA
            </button>
          </div>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
        {displayStats.map((stat, idx) => (
          <div
            key={idx}
            className="bg-[#1B2328] border border-[#2E3A41] rounded-xl p-5 sm:p-6 transition-all hover:border-[#8DA0A8]/50"
          >
            <div className="font-mono text-2xl sm:text-3xl font-bold text-[#E9EDEE]">
              {stat.value}
            </div>
            <div className="text-xs sm:text-sm text-[#8DA0A8] mt-1 font-medium">
              {stat.label}
            </div>
          </div>
        ))}
      </div>

      {/* Summary highlight cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-[#1B2328] border border-[#2E3A41] rounded-xl p-6 space-y-3">
          <div className="w-10 h-10 rounded-lg bg-[#5FD3A0]/10 border border-[#5FD3A0]/30 flex items-center justify-center text-[#5FD3A0]">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <h3 className="text-lg font-bold text-[#E9EDEE]">Transformer Power</h3>
          <p className="text-sm text-[#8DA0A8] leading-relaxed">
            DistilBERT fine-tuning delivers an outstanding 98.05% accuracy, detecting subtle GPT-2 generation signatures across 10 Amazon categories.
          </p>
        </div>

        <div className="bg-[#1B2328] border border-[#2E3A41] rounded-xl p-6 space-y-3">
          <div className="w-10 h-10 rounded-lg bg-[#F4C95D]/10 border border-[#F4C95D]/30 flex items-center justify-center text-[#F4C95D]">
            <Layers className="w-5 h-5" />
          </div>
          <h3 className="text-lg font-bold text-[#E9EDEE]">Hybrid Fusion Approach</h3>
          <p className="text-sm text-[#8DA0A8] leading-relaxed">
            Combines 100k TF-IDF n-grams with 10 hand-engineered linguistic signals (lexical diversity, punctuation ratios, sentiment scores) reaching 90.89% accuracy.
          </p>
        </div>

        <div className="bg-[#1B2328] border border-[#2E3A41] rounded-xl p-6 space-y-3">
          <div className="w-10 h-10 rounded-lg bg-[#E64980]/10 border border-[#E64980]/30 flex items-center justify-center text-[#E64980]">
            <Zap className="w-5 h-5" />
          </div>
          <h3 className="text-lg font-bold text-[#E9EDEE]">XAI & Error Analysis</h3>
          <p className="text-sm text-[#8DA0A8] leading-relaxed">
            Integrated SHAP token contribution charts explain exact model reasoning for every review, paired with error analysis on false positives and negatives.
          </p>
        </div>
      </div>

      {/* Info Notice */}
      <div className="bg-[#212B31] border border-[#2E3A41] rounded-xl p-5 flex items-center justify-between text-xs sm:text-sm text-[#8DA0A8]">
        <span>Capstone Project in Data Science II (DS4105) - Sabaragamuwa University of Sri Lanka</span>
        <span className="font-mono text-[#5FD3A0]">Verified Research Pipeline</span>
      </div>
    </div>
  );
}
