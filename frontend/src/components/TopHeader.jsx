import React from 'react';
import { Menu, ShieldCheck, Activity, Search, Database, Cpu, CheckCircle2, FileText, Link2, Sparkles } from 'lucide-react';

export default function TopHeader({ activeTab, setIsMobileOpen, health }) {
  const metaMap = {
    Overview: {
      title: 'Dashboard Overview',
      subtitle: 'Overview of analytics, dataset stats, and model performance',
      icon: Activity,
    },
    Dataset: {
      title: 'The Dataset',
      subtitle: '40,405 Amazon reviews, 10 categories, balanced real vs. fake split',
      icon: Database,
    },
    Investigation: {
      title: 'The Investigation',
      subtitle: 'Benchmark comparison across four modeling approaches',
      icon: Cpu,
    },
    Verdict: {
      title: 'The Verdict',
      subtitle: 'DistilBERT test set evaluation, metrics, and confusion matrix',
      icon: CheckCircle2,
    },
    CaseNotes: {
      title: 'Case Notes',
      subtitle: 'Error analysis of false positives and false negatives',
      icon: FileText,
    },
    LiveInterrogation: {
      title: 'Live Interrogation',
      subtitle: 'Test custom review text with BERT transformer & SHAP explanation',
      icon: Search,
    },
    UrlScraper: {
      title: 'URL Scraper',
      subtitle: 'Scrape e-commerce product pages and calculate Product Trust Scores',
      icon: Link2,
    },
  };

  const currentMeta = metaMap[activeTab] || metaMap.Overview;
  const ActiveIcon = currentMeta.icon;

  return (
    <header className="sticky top-0 z-30 bg-white/90 backdrop-blur-md border-b border-slate-200 px-4 sm:px-6 py-4">
      <div className="max-w-[1600px] mx-auto flex items-center justify-between gap-4">
        {/* Left: Mobile hamburger & Page Title */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => setIsMobileOpen(true)}
            className="md:hidden p-2 rounded-lg text-slate-500 hover:text-slate-800 hover:bg-slate-100 transition-colors"
            aria-label="Open Navigation Menu"
          >
            <Menu className="w-5 h-5" />
          </button>

          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold text-indigo-600 bg-indigo-50 px-2.5 py-0.5 rounded-full border border-indigo-100 flex items-center gap-1.5">
                <ActiveIcon className="w-3.5 h-3.5 text-indigo-600" />
                {activeTab}
              </span>
              <span className="hidden sm:inline text-xs text-slate-400">•</span>
              <span className="hidden sm:inline text-xs font-mono text-slate-500">
                DS4105 CAPSTONE
              </span>
            </div>
            <h2 className="text-lg sm:text-xl font-bold text-slate-900 tracking-tight mt-0.5">
              {currentMeta.title}
            </h2>
          </div>
        </div>

        {/* Right: Status Badges & Helpful Info */}
        <div className="flex items-center gap-2.5">
          <div className="hidden sm:flex items-center gap-2 bg-slate-50 border border-slate-200 px-3 py-1.5 rounded-lg text-xs font-mono">
            <span
              className={`w-2 h-2 rounded-full ${
                health?.status === 'online' ? 'bg-emerald-500' : 'bg-rose-500'
              }`}
            ></span>
            <span className="text-slate-600 font-medium">
              API: <strong className="text-slate-900">{health?.status === 'online' ? 'Connected' : 'Offline'}</strong>
            </span>
          </div>

          {health?.bert_model_available && (
            <span className="hidden lg:inline-flex items-center gap-1 bg-emerald-50 border border-emerald-200 text-emerald-700 text-xs font-mono px-2.5 py-1 rounded-lg font-semibold">
              <Sparkles className="w-3 h-3 text-emerald-600" />
              BERT Ready
            </span>
          )}
        </div>
      </div>
    </header>
  );
}
