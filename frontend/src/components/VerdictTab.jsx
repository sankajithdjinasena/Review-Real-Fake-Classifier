import React from 'react';
import { Award, CheckCircle2, AlertTriangle, ShieldCheck, Target, Percent } from 'lucide-react';

export default function VerdictTab({ data }) {
  const metrics = data?.metrics || {
    accuracy: '98.05%',
    f1: '98.07%',
    precision: '97.03%',
    recall: '99.14%',
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div>
        <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-indigo-50 border border-indigo-100 text-indigo-700 text-xs font-semibold mb-2">
          <CheckCircle2 className="w-3.5 h-3.5 text-indigo-600" />
          03 - THE VERDICT & FINAL EVALUATION
        </div>
        <h2 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
          DistilBERT evaluated on 6,061 held-out reviews
        </h2>
        <p className="text-sm text-slate-500 mt-1">
          Final test set performance metrics on unseen real Amazon reviews and GPT-2 fake review evaluation split.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Confusion Matrix Visualizer */}
        <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-2xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
                <Award className="w-5 h-5 text-indigo-600" />
                Confusion Matrix Heatmap
              </h3>
              <span className="text-xs font-mono text-slate-400">N = 6,061 Test Samples</span>
            </div>

            <div className="grid grid-cols-2 gap-4 text-center font-mono">
              <div className="bg-emerald-50/80 border border-emerald-200 rounded-xl p-6 flex flex-col items-center justify-center">
                <span className="text-3xl sm:text-4xl font-extrabold text-emerald-700">2,941</span>
                <span className="text-xs text-slate-600 uppercase tracking-wider mt-2 font-sans font-semibold">True Real</span>
                <span className="text-[11px] text-emerald-800/80 mt-1">Actual Real → Pred Real</span>
              </div>

              <div className="bg-rose-50/80 border border-rose-200 rounded-xl p-6 flex flex-col items-center justify-center">
                <span className="text-3xl sm:text-4xl font-extrabold text-rose-700">92</span>
                <span className="text-xs text-slate-600 uppercase tracking-wider mt-2 font-sans font-semibold">False Positive</span>
                <span className="text-[11px] text-rose-800/80 mt-1">Actual Real → Pred Fake</span>
              </div>

              <div className="bg-amber-50/80 border border-amber-200 rounded-xl p-6 flex flex-col items-center justify-center">
                <span className="text-3xl sm:text-4xl font-extrabold text-amber-800">26</span>
                <span className="text-xs text-slate-600 uppercase tracking-wider mt-2 font-sans font-semibold">False Negative</span>
                <span className="text-[11px] text-amber-800/80 mt-1">Actual Fake → Pred Real</span>
              </div>

              <div className="bg-indigo-50/80 border border-indigo-200 rounded-xl p-6 flex flex-col items-center justify-center">
                <span className="text-3xl sm:text-4xl font-extrabold text-indigo-700">3,002</span>
                <span className="text-xs text-slate-600 uppercase tracking-wider mt-2 font-sans font-semibold">True Fake</span>
                <span className="text-[11px] text-indigo-800/80 mt-1">Actual Fake → Pred Fake</span>
              </div>
            </div>
          </div>
          <div className="text-xs text-slate-500 mt-6 pt-4 border-t border-slate-100 flex items-center justify-between font-mono">
            <span>Evaluated Test Split: 6,061 reviews</span>
            <span className="text-emerald-700 font-bold">Overall Accuracy: 98.05%</span>
          </div>
        </div>

        {/* Test Set Metrics Grid */}
        <div className="space-y-6 flex flex-col justify-between">
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-2xs">
              <span className="text-xs text-slate-500 font-semibold block mb-1">Accuracy</span>
              <div className="text-3xl font-bold font-mono text-indigo-600">{metrics.accuracy}</div>
              <span className="text-[11px] text-slate-400 mt-1 block">Overall classification accuracy</span>
            </div>
            <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-2xs">
              <span className="text-xs text-slate-500 font-semibold block mb-1">F1 Score (Fake)</span>
              <div className="text-3xl font-bold font-mono text-emerald-600">{metrics.f1}</div>
              <span className="text-[11px] text-slate-400 mt-1 block">Harmonic mean of P & R</span>
            </div>
            <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-2xs">
              <span className="text-xs text-slate-500 font-semibold block mb-1">Precision (Fake)</span>
              <div className="text-3xl font-bold font-mono text-slate-900">{metrics.precision}</div>
              <span className="text-[11px] text-slate-400 mt-1 block">Low false alarm rate</span>
            </div>
            <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-2xs">
              <span className="text-xs text-slate-500 font-semibold block mb-1">Recall (Fake)</span>
              <div className="text-3xl font-bold font-mono text-slate-900">{metrics.recall}</div>
              <span className="text-[11px] text-slate-400 mt-1 block">99.14% fake reviews caught</span>
            </div>
          </div>

          <div className="bg-amber-50/80 border border-amber-200 rounded-xl p-5 space-y-2.5 text-amber-900">
            <div className="flex items-center gap-2 font-bold text-sm text-amber-800">
              <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
              Critical Note on Test Performance
            </div>
            <p className="text-xs text-amber-800/90 leading-relaxed">
              A ~7-point accuracy gain over the best classical fusion model (90.89%) - but likely influenced by GPT-2's comparatively easy-to-detect synthetic generation artifacts. Review the <strong>Case Notes</strong> tab to examine specific error failure modes.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
