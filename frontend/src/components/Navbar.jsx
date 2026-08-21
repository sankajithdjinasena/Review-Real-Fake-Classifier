import React from 'react';
import { Search, Database, Cpu, CheckCircle2, FileText, Activity } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab, health }) {
  const navItems = [
    { id: 'Overview', label: 'Overview', icon: Activity },
    { id: 'Dataset', label: 'The Dataset', icon: Database },
    { id: 'Investigation', label: 'The Investigation', icon: Cpu },
    { id: 'Verdict', label: 'The Verdict', icon: CheckCircle2 },
    { id: 'CaseNotes', label: 'Case Notes', icon: FileText },
    { id: 'LiveInterrogation', label: '🔎 Live Interrogation', icon: Search },
  ];

  return (
    <header className="bg-[#1B2328] border-b border-[#2E3A41] sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Top row */}
        <div className="py-4 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <div className="inline-flex items-center gap-2 font-mono text-xs text-[#8DA0A8] border border-[#2E3A41] px-2.5 py-1 rounded bg-[#12181C]">
              <span className="w-2 h-2 rounded-full bg-[#5FD3A0] animate-pulse"></span>
              CASE FILE - DS4105 CAPSTONE PROJECT
            </div>
            <h1 className="text-xl sm:text-2xl font-extrabold text-[#E9EDEE] tracking-tight mt-1">
              Detecting Deception in Product Reviews
            </h1>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 bg-[#12181C] border border-[#2E3A41] px-3 py-1.5 rounded-lg text-xs font-mono">
              <span className={`w-2.5 h-2.5 rounded-full ${health?.status === 'online' ? 'bg-[#5FD3A0]' : 'bg-[#E64980]'}`}></span>
              <span className="text-[#8DA0A8]">
                API: <strong className="text-[#E9EDEE]">{health?.status === 'online' ? 'Connected' : 'Offline'}</strong>
              </span>
            </div>
            {health?.bert_model_available && (
              <span className="hidden sm:inline-block bg-[#5FD3A0]/10 border border-[#5FD3A0]/30 text-[#5FD3A0] text-xs font-mono px-2.5 py-1 rounded-md">
                BERT Ready
              </span>
            )}
            {health?.fusion_model_available && (
              <span className="hidden sm:inline-block bg-[#F4C95D]/10 border border-[#F4C95D]/30 text-[#F4C95D] text-xs font-mono px-2.5 py-1 rounded-md">
                Fusion Ready
              </span>
            )}
          </div>
        </div>

        {/* Tab Navigation */}
        <nav className="flex space-x-1 sm:space-x-2 overflow-x-auto py-2 scrollbar-none">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center gap-2 px-3.5 py-2 rounded-lg font-medium text-xs sm:text-sm whitespace-nowrap transition-all ${
                  isActive
                    ? 'bg-[#E9EDEE] text-[#12181C] font-semibold shadow'
                    : 'text-[#8DA0A8] hover:text-[#E9EDEE] hover:bg-[#212B31]'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-[#12181C]' : 'text-[#8DA0A8]'}`} />
                {item.label}
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
