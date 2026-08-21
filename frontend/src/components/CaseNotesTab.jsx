import React, { useState } from 'react';
import { AlertCircle, AlertOctagon } from 'lucide-react';

export default function CaseNotesTab({ data }) {
  const [activeSubTab, setActiveSubTab] = useState('fp');

  const falsePositives = data?.false_positives || [];
  const falseNegatives = data?.false_negatives || [];

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div>
        <div className="inline-flex items-center gap-2 font-mono text-xs text-[#8DA0A8] border border-[#2E3A41] px-2.5 py-1 rounded bg-[#1B2328] mb-2">
          <span className="w-2 h-2 rounded-full bg-[#5FD3A0]"></span>
          04 - CASE NOTES
        </div>
        <h2 className="text-2xl sm:text-3xl font-extrabold text-[#E9EDEE]">
          103 of 6,061 test reviews were misclassified
        </h2>
        <p className="text-sm text-[#8DA0A8] mt-1">
          Reading the actual mistakes reveals what the model learned - and what fools it.
        </p>
      </div>

      {/* Sub-tab selection */}
      <div className="flex border-b border-[#2E3A41] gap-6">
        <button
          onClick={() => setActiveSubTab('fp')}
          className={`pb-3 text-sm font-bold flex items-center gap-2 border-b-2 transition-all ${
            activeSubTab === 'fp'
              ? 'border-[#E64980] text-[#E64980]'
              : 'border-transparent text-[#8DA0A8] hover:text-[#E9EDEE]'
          }`}
        >
          <AlertOctagon className="w-4 h-4" />
          False Positives (70)
        </button>

        <button
          onClick={() => setActiveSubTab('fn')}
          className={`pb-3 text-sm font-bold flex items-center gap-2 border-b-2 transition-all ${
            activeSubTab === 'fn'
              ? 'border-[#F4C95D] text-[#F4C95D]'
              : 'border-transparent text-[#8DA0A8] hover:text-[#E9EDEE]'
          }`}
        >
          <AlertCircle className="w-4 h-4" />
          False Negatives (33)
        </button>
      </div>

      {/* Cards list */}
      <div className="space-y-4">
        {activeSubTab === 'fp'
          ? falsePositives.map((item, idx) => (
              <div key={idx} className="bg-[#1B2328] border border-[#2E3A41] rounded-xl p-5 sm:p-6 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-bold tracking-wider px-2.5 py-1 rounded border border-[#E64980] text-[#E64980] bg-[#E64980]/10">
                    REAL FLAGGED AS FAKE
                  </span>
                  <span className="text-xs font-mono text-[#8DA0A8]">Exhibit #{idx + 1}</span>
                </div>

                <div className="text-sm sm:text-base text-[#E9EDEE] italic pl-4 border-l-2 border-[#2E3A41] py-1">
                  &ldquo;{item.quote}&rdquo;
                </div>

                <div className="text-xs sm:text-sm text-[#8DA0A8] pt-2">
                  <strong className="text-[#E9EDEE] font-mono">Why it fooled the model:</strong> {item.note}
                </div>
              </div>
            ))
          : falseNegatives.map((item, idx) => (
              <div key={idx} className="bg-[#1B2328] border border-[#2E3A41] rounded-xl p-5 sm:p-6 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-bold tracking-wider px-2.5 py-1 rounded border border-[#F4C95D] text-[#F4C95D] bg-[#F4C95D]/10">
                    FAKE MISSED AS REAL
                  </span>
                  <span className="text-xs font-mono text-[#8DA0A8]">Exhibit #{idx + 1}</span>
                </div>

                <div className="text-sm sm:text-base text-[#E9EDEE] italic pl-4 border-l-2 border-[#2E3A41] py-1">
                  &ldquo;{item.quote}&rdquo;
                </div>

                <div className="text-xs sm:text-sm text-[#8DA0A8] pt-2">
                  <strong className="text-[#E9EDEE] font-mono">Why it slipped through:</strong> {item.note}
                </div>
              </div>
            ))}
      </div>
    </div>
  );
}
