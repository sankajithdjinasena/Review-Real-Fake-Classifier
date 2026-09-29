import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import TopHeader from './components/TopHeader';
import OverviewTab from './components/OverviewTab';
import DatasetTab from './components/DatasetTab';
import InvestigationTab from './components/InvestigationTab';
import VerdictTab from './components/VerdictTab';
import CaseNotesTab from './components/CaseNotesTab';
import LiveInterrogationTab from './components/LiveInterrogationTab';
import UrlScraperTab from './components/UrlScraperTab';

export default function App() {
  const [activeTab, setActiveTab] = useState('Overview');
  const [health, setHealth] = useState(null);
  const [stats, setStats] = useState(null);
  const [datasetData, setDatasetData] = useState(null);
  const [investigationData, setInvestigationData] = useState(null);
  const [verdictData, setVerdictData] = useState(null);
  const [caseNotesData, setCaseNotesData] = useState(null);

  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [isMobileOpen, setIsMobileOpen] = useState(false);

  useEffect(() => {
    // Fetch initial status & static dashboard data
    fetch('/api/health')
      .then((res) => res.json())
      .then((d) => setHealth(d))
      .catch((err) => console.log('Health check failed:', err));

    fetch('/api/stats')
      .then((res) => res.json())
      .then((d) => setStats(d.stats))
      .catch((err) => console.log('Stats check failed:', err));

    fetch('/api/dataset')
      .then((res) => res.json())
      .then((d) => setDatasetData(d))
      .catch((err) => console.log('Dataset check failed:', err));

    fetch('/api/investigation')
      .then((res) => res.json())
      .then((d) => setInvestigationData(d))
      .catch((err) => console.log('Investigation check failed:', err));

    fetch('/api/verdict')
      .then((res) => res.json())
      .then((d) => setVerdictData(d))
      .catch((err) => console.log('Verdict check failed:', err));

    fetch('/api/case-notes')
      .then((res) => res.json())
      .then((d) => setCaseNotesData(d))
      .catch((err) => console.log('Case notes check failed:', err));
  }, []);

  return (
    <div className="min-h-screen bg-[#F8FAFC] text-[#0F172A] flex flex-col font-sans selection:bg-indigo-100 selection:text-indigo-700">
      {/* Sidebar Navigation */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        health={health}
        isCollapsed={isSidebarCollapsed}
        setIsCollapsed={setIsSidebarCollapsed}
        isMobileOpen={isMobileOpen}
        setIsMobileOpen={setIsMobileOpen}
      />

      {/* Main Layout Area */}
      <div
        className={`flex-1 flex flex-col transition-all duration-300 ${
          isSidebarCollapsed ? 'md:pl-[80px]' : 'md:pl-[260px]'
        }`}
      >
        {/* Top Header */}
        <TopHeader
          activeTab={activeTab}
          setIsMobileOpen={setIsMobileOpen}
          health={health}
        />

        {/* Workspace Content */}
        <main className="flex-1 p-4 sm:p-6 lg:p-8 max-w-[1600px] w-full mx-auto">
          {activeTab === 'Overview' && (
            <OverviewTab stats={stats} setActiveTab={setActiveTab} />
          )}
          {activeTab === 'Dataset' && <DatasetTab data={datasetData} />}
          {activeTab === 'Investigation' && (
            <InvestigationTab data={investigationData} />
          )}
          {activeTab === 'Verdict' && <VerdictTab data={verdictData} />}
          {activeTab === 'CaseNotes' && <CaseNotesTab data={caseNotesData} />}
          {activeTab === 'LiveInterrogation' && (
            <LiveInterrogationTab health={health} />
          )}
          {activeTab === 'UrlScraper' && <UrlScraperTab />}
        </main>

        {/* Footer */}
        <footer className="border-t border-slate-200 py-5 bg-white text-slate-500 text-xs">
          <div className="max-w-[1600px] mx-auto px-4 sm:px-6 flex flex-col sm:flex-row justify-between items-center gap-3 font-mono">
            <span>
              Capstone Project in Data Science II (DS4105) - Sabaragamuwa University of Sri Lanka
            </span>
            <span className="text-emerald-700 bg-emerald-50 px-2.5 py-1 rounded border border-emerald-200 font-semibold">
              React 18 + FastAPI Powered
            </span>
          </div>
        </footer>
      </div>
    </div>
  );
}
