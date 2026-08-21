import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import OverviewTab from './components/OverviewTab';
import DatasetTab from './components/DatasetTab';
import InvestigationTab from './components/InvestigationTab';
import VerdictTab from './components/VerdictTab';
import CaseNotesTab from './components/CaseNotesTab';
import LiveInterrogationTab from './components/LiveInterrogationTab';

export default function App() {
  const [activeTab, setActiveTab] = useState('Overview');
  const [health, setHealth] = useState(null);
  const [stats, setStats] = useState(null);
  const [datasetData, setDatasetData] = useState(null);
  const [investigationData, setInvestigationData] = useState(null);
  const [verdictData, setVerdictData] = useState(null);
  const [caseNotesData, setCaseNotesData] = useState(null);

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
    <div className="min-h-screen bg-[#12181C] text-[#E9EDEE] flex flex-col font-sans">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} health={health} />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
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
      </main>

      <footer className="border-t border-[#2E3A41] py-6 bg-[#1B2328] text-center text-xs text-[#8DA0A8]">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row justify-between items-center gap-2 font-mono">
          <span>Capstone Project in Data Science II (DS4105) - Sabaragamuwa University of Sri Lanka</span>
          <span className="text-[#5FD3A0]">React 18 + FastAPI Powered</span>
        </div>
      </footer>
    </div>
  );
}
