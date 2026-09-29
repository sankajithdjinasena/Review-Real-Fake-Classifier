import React from 'react';
import {
  Activity,
  Database,
  Cpu,
  CheckCircle2,
  FileText,
  Search,
  Link2,
  ChevronLeft,
  ChevronRight,
  ShieldCheck,
  Sparkles,
  X,
  HelpCircle,
  BarChart2
} from 'lucide-react';

export default function Sidebar({
  activeTab,
  setActiveTab,
  health,
  isCollapsed,
  setIsCollapsed,
  isMobileOpen,
  setIsMobileOpen,
}) {
  const navItems = [
    { id: 'Overview', label: 'Overview', icon: Activity, section: 'Main' },
    { id: 'Dataset', label: 'The Dataset', icon: Database, section: 'Analytics' },
    { id: 'Investigation', label: 'The Investigation', icon: Cpu, section: 'Analytics' },
    { id: 'Verdict', label: 'The Verdict', icon: CheckCircle2, section: 'Analytics' },
    { id: 'CaseNotes', label: 'Case Notes', icon: FileText, section: 'Analytics' },
    { id: 'LiveInterrogation', label: 'Live Interrogation', icon: Search, section: 'Tools' },
    { id: 'UrlScraper', label: 'URL Scraper', icon: Link2, section: 'Tools' },
  ];

  const renderNavContent = (collapsed = false) => (
    <div className="flex flex-col h-full justify-between py-4">
      {/* Top Section: Logo & Nav List */}
      <div className="space-y-6">
        {/* Header Logo */}
        <div className={`px-4 flex items-center ${collapsed ? 'justify-center' : 'justify-between'}`}>
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-indigo-600 flex items-center justify-center text-white shadow-sm shrink-0">
              <ShieldCheck className="w-5 h-5" />
            </div>
            {!collapsed && (
              <div className="truncate">
                <h1 className="font-bold text-slate-900 text-sm leading-tight tracking-tight">
                  Review Classifier
                </h1>
                <p className="text-[11px] font-mono text-slate-500 font-medium truncate">
                  DS4105 Capstone
                </p>
              </div>
            )}
          </div>

          {/* Desktop collapse toggle button */}
          {!collapsed && (
            <button
              onClick={() => setIsCollapsed(!isCollapsed)}
              className="hidden md:flex p-1.5 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors"
              title="Collapse Sidebar"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Navigation Items */}
        <nav className="px-3 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => {
                  setActiveTab(item.id);
                  setIsMobileOpen(false);
                }}
                title={collapsed ? item.label : undefined}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-xs font-semibold transition-all duration-150 ${
                  isActive
                    ? 'bg-indigo-50 text-indigo-600 font-bold'
                    : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
                } ${collapsed ? 'justify-center px-0' : ''}`}
              >
                <Icon
                  className={`w-4 h-4 shrink-0 transition-colors ${
                    isActive ? 'text-indigo-600' : 'text-slate-400 group-hover:text-slate-600'
                  }`}
                />
                {!collapsed && <span className="truncate">{item.label}</span>}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Bottom Section: API Status & Footer */}
      <div className="px-3 space-y-3">
        {/* Status Indicator Card */}
        {!collapsed ? (
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-3 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-500 font-medium">Backend API</span>
              <span className="flex items-center gap-1.5 font-mono text-[11px] font-bold">
                <span
                  className={`w-2 h-2 rounded-full ${
                    health?.status === 'online' ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'
                  }`}
                ></span>
                <span className={health?.status === 'online' ? 'text-emerald-700' : 'text-rose-700'}>
                  {health?.status === 'online' ? 'Connected' : 'Offline'}
                </span>
              </span>
            </div>

            <div className="flex flex-wrap gap-1.5 pt-1">
              {health?.bert_model_available && (
                <span className="bg-emerald-50 border border-emerald-200 text-emerald-700 text-[10px] font-mono px-2 py-0.5 rounded font-semibold">
                  BERT Ready
                </span>
              )}
              {health?.fusion_model_available && (
                <span className="bg-amber-50 border border-amber-200 text-amber-700 text-[10px] font-mono px-2 py-0.5 rounded font-semibold">
                  Fusion Ready
                </span>
              )}
            </div>
          </div>
        ) : (
          <div className="flex justify-center" title={`API: ${health?.status === 'online' ? 'Connected' : 'Offline'}`}>
            <span
              className={`w-3 h-3 rounded-full ${
                health?.status === 'online' ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'
              }`}
            ></span>
          </div>
        )}

        {/* Expand Sidebar toggle when collapsed */}
        {collapsed && (
          <button
            onClick={() => setIsCollapsed(false)}
            className="w-full flex justify-center py-2 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
            title="Expand Sidebar"
          >
            <ChevronRight className="w-5 h-5" />
          </button>
        )}
      </div>
    </div>
  );

  return (
    <>
      {/* Desktop Sidebar (Fixed) */}
      <aside
        className={`hidden md:block fixed top-0 left-0 bottom-0 z-40 bg-white border-r border-slate-200 transition-all duration-300 ${
          isCollapsed ? 'w-[80px]' : 'w-[260px]'
        }`}
      >
        {renderNavContent(isCollapsed)}
      </aside>

      {/* Mobile Drawer Backdrop & Menu */}
      {isMobileOpen && (
        <div className="md:hidden fixed inset-0 z-50 flex">
          {/* Backdrop */}
          <div
            className="fixed inset-0 bg-slate-900/40 backdrop-blur-xs transition-opacity"
            onClick={() => setIsMobileOpen(false)}
          ></div>

          {/* Drawer Sidebar */}
          <div className="relative w-[270px] max-w-[80vw] bg-white h-full shadow-2xl flex flex-col z-10 animate-fadeIn">
            <button
              onClick={() => setIsMobileOpen(false)}
              className="absolute top-4 right-4 p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100"
            >
              <X className="w-5 h-5" />
            </button>
            {renderNavContent(false)}
          </div>
        </div>
      )}
    </>
  );
}
