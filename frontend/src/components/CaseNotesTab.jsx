import React, { useState } from 'react';
import { AlertCircle, AlertOctagon, FileText, Info } from 'lucide-react';

export default function CaseNotesTab({ data }) {
  const [activeSubTab, setActiveSubTab] = useState('fp');

  const falsePositives = data?.false_positives || [];
  const falseNegatives = data?.false_negatives || [];

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div>
        <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-indigo-50 border border-indigo-100 text-indigo-700 text-xs font-semibold mb-2">
          <FileText className="w-3.5 h-3.5 text-indigo-600" />
          04 - CASE NOTES & ERROR ANALYSIS
        </div>
        <h2 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
          103 of 6,061 test reviews were misclassified
        </h2>
        <p className="text-sm text-slate-500 mt-1">
          Analyzing actual prediction mistakes reveals what patterns the model learned - and what tricks fool it.
        </p>
      </div>

      {/* Explanatory Banner */}
      <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 sm:p-5 flex items-start gap-3 text-xs sm:text-sm text-slate-600">
        <Info className="w-4 h-4 text-indigo-600 shrink-0 mt-0.5" />
        <div>
          <strong className="text-slate-900">Why 10 representative case exhibits are featured:</strong> Out of 6,061 held-out test reviews, the fine-tuned model made 103 total misclassifications (70 False Positives and 33 False Negatives). Below are the <strong>10 curated representative case studies</strong> (5 per error class) selected for in-depth qualitative error analysis to explain the model's primary failure modes.
        </div>
      </div>

      {/* Sub-tab selection */}
      <div className="flex border-b border-slate-200 gap-4">
        <button
          onClick={() => setActiveSubTab('fp')}
          className={`pb-3 text-sm font-bold flex items-center gap-2 border-b-2 transition-all cursor-pointer ${
            activeSubTab === 'fp'
              ? 'border-rose-600 text-rose-600'
              : 'border-transparent text-slate-500 hover:text-slate-900'
          }`}
        >
          <AlertOctagon className="w-4 h-4" />
          False Positives ({falsePositives.length} Exhibits • 70 Total Errors)
        </button>

        <button
          onClick={() => setActiveSubTab('fn')}
          className={`pb-3 text-sm font-bold flex items-center gap-2 border-b-2 transition-all cursor-pointer ${
            activeSubTab === 'fn'
              ? 'border-amber-600 text-amber-700'
              : 'border-transparent text-slate-500 hover:text-slate-900'
          }`}
        >
          <AlertCircle className="w-4 h-4" />
          False Negatives ({falseNegatives.length} Exhibits • 33 Total Errors)
        </button>
      </div>

      {/* Cards list */}
      <div className="space-y-4">
        {activeSubTab === 'fp'
          ? falsePositives.map((item, idx) => (
              <div key={idx} className="bg-white border border-slate-200 rounded-xl p-5 sm:p-6 space-y-3.5 shadow-2xs">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-bold tracking-wider px-2.5 py-1 rounded border border-rose-200 text-rose-700 bg-rose-50">
                    REAL FLAGGED AS FAKE
                  </span>
                  <span className="text-xs font-mono text-slate-400">Exhibit #{idx + 1} of {falsePositives.length}</span>
                </div>

                <div className="text-sm sm:text-base text-slate-800 italic bg-slate-50 border-l-4 border-rose-500 rounded-r-lg p-4">
                  &ldquo;{item.quote}&rdquo;
                </div>

                <div className="text-xs sm:text-sm text-slate-600 pt-1">
                  <strong className="text-slate-900 font-mono">Why it fooled the model:</strong> {item.note}
                </div>
              </div>
            ))
          : falseNegatives.map((item, idx) => (
              <div key={idx} className="bg-white border border-slate-200 rounded-xl p-5 sm:p-6 space-y-3.5 shadow-2xs">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-bold tracking-wider px-2.5 py-1 rounded border border-amber-200 text-amber-800 bg-amber-50">
                    FAKE MISSED AS REAL
                  </span>
                  <span className="text-xs font-mono text-slate-400">Exhibit #{idx + 1} of {falseNegatives.length}</span>
                </div>

                <div className="text-sm sm:text-base text-slate-800 italic bg-slate-50 border-l-4 border-amber-500 rounded-r-lg p-4">
                  &ldquo;{item.quote}&rdquo;
                </div>

                <div className="text-xs sm:text-sm text-slate-600 pt-1">
                  <strong className="text-slate-900 font-mono">Why it slipped through:</strong> {item.note}
                </div>
              </div>
            ))}
      </div>
    </div>
  );
}
