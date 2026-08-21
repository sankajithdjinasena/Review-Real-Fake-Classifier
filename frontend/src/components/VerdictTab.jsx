import React from 'react';
import { Award, CheckCircle2, AlertTriangle } from 'lucide-react';

export default function VerdictTab({ data }) {
  const metrics = data?.metrics || {
    accuracy: "98.30%",
    f1: "98.31%",
    precision: "97.72%",
    recall: "98.91%",
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div>
        <div className="inline-flex items-center gap-2 font-mono text-xs text-[#8DA0A8] border border-[#2E3A41] px-2.5 py-1 rounded bg-[#1B2328] mb-2">
          <span className="w-2 h-2 rounded-full bg-[#5FD3A0]"></span>
          03 - THE VERDICT
        </div>
        <h2 className="text-2xl sm:text-3xl font-extrabold text-[#E9EDEE]">
          DistilBERT, evaluated on 6,061 held-out reviews
        </h2>
        <p className="text-sm text-[#8DA0A8] mt-1">
          Final model test set performance on unseen Amazon & GPT-2 review evaluation split.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Confusion Matrix Visualizer */}
        <div className="bg-[#1B2328] border border-[#2E3A41] rounded-xl p-6 flex flex-col justify-between">
          <div>
            <h3 className="text-base font-bold text-[#E9EDEE] mb-6 flex items-center gap-2">
              <Award className="w-5 h-5 text-[#5FD3A0]" />
              Confusion Matrix Heatmap
            </h3>
            <div className="grid grid-cols-2 gap-4 text-center font-mono">
              <div className="bg-[#5FD3A0]/15 border border-[#5FD3A0]/40 rounded-xl p-6 flex flex-col items-center justify-center">
                <span className="text-3xl sm:text-4xl font-extrabold text-[#5FD3A0]">2,963</span>
                <span className="text-xs text-[#8DA0A8] uppercase tracking-wider mt-2 font-sans font-semibold">True Real</span>
                <span className="text-[11px] text-[#5FD3A0]/80 mt-1">Actual Real → Pred Real</span>
              </div>

              <div className="bg-[#E64980]/15 border border-[#E64980]/40 rounded-xl p-6 flex flex-col items-center justify-center">
                <span className="text-3xl sm:text-4xl font-extrabold text-[#E64980]">70</span>
                <span className="text-xs text-[#8DA0A8] uppercase tracking-wider mt-2 font-sans font-semibold">False Positive</span>
                <span className="text-[11px] text-[#E64980]/80 mt-1">Actual Real → Pred Fake</span>
              </div>

              <div className="bg-[#F4C95D]/15 border border-[#F4C95D]/40 rounded-xl p-6 flex flex-col items-center justify-center">
                <span className="text-3xl sm:text-4xl font-extrabold text-[#F4C95D]">33</span>
                <span className="text-xs text-[#8DA0A8] uppercase tracking-wider mt-2 font-sans font-semibold">False Negative</span>
                <span className="text-[11px] text-[#F4C95D]/80 mt-1">Actual Fake → Pred Real</span>
              </div>

              <div className="bg-[#5FD3A0]/15 border border-[#5FD3A0]/40 rounded-xl p-6 flex flex-col items-center justify-center">
                <span className="text-3xl sm:text-4xl font-extrabold text-[#5FD3A0]">2,995</span>
                <span className="text-xs text-[#8DA0A8] uppercase tracking-wider mt-2 font-sans font-semibold">True Fake</span>
                <span className="text-[11px] text-[#5FD3A0]/80 mt-1">Actual Fake → Pred Fake</span>
              </div>
            </div>
          </div>
          <div className="text-xs text-[#8DA0A8] mt-6 pt-4 border-t border-[#2E3A41] flex items-center justify-between">
            <span>Total Evaluated: 6,061 reviews</span>
            <span className="text-[#5FD3A0]">Overall Accuracy: 98.30%</span>
          </div>
        </div>

        {/* Test Set Metrics Grid */}
        <div className="space-y-6 flex flex-col justify-between">
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-[#1B2328] border border-[#2E3A41] rounded-xl p-5">
              <div className="text-3xl font-bold font-mono text-[#5FD3A0]">{metrics.accuracy}</div>
              <div className="text-xs text-[#8DA0A8] mt-1">Accuracy</div>
            </div>
            <div className="bg-[#1B2328] border border-[#2E3A41] rounded-xl p-5">
              <div className="text-3xl font-bold font-mono text-[#5FD3A0]">{metrics.f1}</div>
              <div className="text-xs text-[#8DA0A8] mt-1">F1 Score (Fake Class)</div>
            </div>
            <div className="bg-[#1B2328] border border-[#2E3A41] rounded-xl p-5">
              <div className="text-3xl font-bold font-mono text-[#5FD3A0]">{metrics.precision}</div>
              <div className="text-xs text-[#8DA0A8] mt-1">Precision (Fake)</div>
            </div>
            <div className="bg-[#1B2328] border border-[#2E3A41] rounded-xl p-5">
              <div className="text-3xl font-bold font-mono text-[#5FD3A0]">{metrics.recall}</div>
              <div className="text-xs text-[#8DA0A8] mt-1">Recall (Fake)</div>
            </div>
          </div>

          <div className="bg-[#212B31] border border-[#2E3A41] rounded-xl p-5 space-y-3">
            <div className="flex items-center gap-2 text-[#F4C95D] font-bold text-sm">
              <AlertTriangle className="w-4 h-4" />
              Critical Note on Performance
            </div>
            <p className="text-xs text-[#8DA0A8] leading-relaxed">
              A ~8-point accuracy gain over the best classical fusion model (90.36%) - but likely inflated by GPT-2's comparatively easy-to-detect generation artifacts. Review the Case Notes tab to see exact error failure modes.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
