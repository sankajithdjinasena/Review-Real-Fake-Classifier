import React from 'react';
import {
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend,
  BarChart, Bar, XAxis, YAxis, CartesianGrid
} from 'recharts';
import { Database, AlertTriangle, Layers, BarChart3, FileText, CheckCircle2 } from 'lucide-react';

export default function DatasetTab({ data }) {
  const classBalance = data?.class_balance || [
    { name: 'Real', value: 20215, color: '#059669' },
    { name: 'Fake', value: 20190, color: '#E11D48' },
  ];

  const reviewLength = data?.review_length || [
    { category: 'Avg. Words per Review', Real: 73.6, Fake: 61.3 }
  ];

  const categories = data?.categories || [
    { name: 'Kindle Store', count: 4700 },
    { name: 'Books', count: 4400 },
    { name: 'Pet Supplies', count: 4250 },
    { name: 'Home & Kitchen', count: 4050 },
    { name: 'Electronics', count: 4000 },
    { name: 'Sports & Outdoors', count: 3950 },
    { name: 'Tools & Home Impr.', count: 3900 },
    { name: 'Clothing/Shoes/Jewelry', count: 3850 },
    { name: 'Toys & Games', count: 3800 },
    { name: 'Movies & TV', count: 3600 },
  ];

  const ratings = data?.rating_distribution || [
    { rating: '1★', Real: 1050, Fake: 1050 },
    { rating: '2★', Real: 950, Fake: 950 },
    { rating: '3★', Real: 1850, Fake: 1950 },
    { rating: '4★', Real: 4050, Fake: 3950 },
    { rating: '5★', Real: 12250, Fake: 12250 },
  ];

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Page Header */}
      <div>
        <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-indigo-50 border border-indigo-100 text-indigo-700 text-xs font-semibold mb-2">
          <Database className="w-3.5 h-3.5 text-indigo-600" />
          01 - DATASET & EXPLORATORY ANALYSIS
        </div>
        <h2 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
          40,405 reviews, near-perfectly balanced
        </h2>
        <p className="text-sm text-slate-500 mt-1">
          10 product categories, real Amazon reviews vs. synthetic GPT-2 generated fakes.
        </p>
      </div>

      {/* KPI Cards Summary Row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-2xs">
          <span className="text-xs font-semibold text-slate-500">Total Review Samples</span>
          <div className="font-mono text-2xl font-bold text-slate-900 mt-1">40,405</div>
          <span className="text-xs text-slate-400 mt-1 block">Full dataset size</span>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-2xs">
          <span className="text-xs font-semibold text-slate-500">Class Balance</span>
          <div className="font-mono text-2xl font-bold text-emerald-600 mt-1">50.03% / 49.97%</div>
          <span className="text-xs text-slate-400 mt-1 block">Near-perfect 1:1 ratio</span>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-2xs">
          <span className="text-xs font-semibold text-slate-500">Real Avg. Length</span>
          <div className="font-mono text-2xl font-bold text-indigo-600 mt-1">73.6 words</div>
          <span className="text-xs text-slate-400 mt-1 block">+12.3 words vs fakes</span>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-2xs">
          <span className="text-xs font-semibold text-slate-500">Product Categories</span>
          <div className="font-mono text-2xl font-bold text-slate-900 mt-1">10 Categories</div>
          <span className="text-xs text-slate-400 mt-1 block">Amazon e-commerce</span>
        </div>
      </div>

      {/* Row 1: Pie & Length Bar Chart */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Class Balance Pie Chart */}
        <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-2xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-base font-bold text-slate-900">Class Distribution</h3>
              <span className="text-xs font-mono text-slate-400">Target Label</span>
            </div>
            <div className="w-full h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={classBalance}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={90}
                    paddingAngle={4}
                    dataKey="value"
                  >
                    {classBalance.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#FFFFFF',
                      borderColor: '#E2E8F0',
                      color: '#0F172A',
                      borderRadius: '8px',
                      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                      fontSize: '12px',
                    }}
                    itemStyle={{ color: '#0F172A', fontWeight: 600 }}
                  />
                  <Legend formatter={(value) => <span className="text-slate-600 text-xs font-medium">{value}</span>} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="flex justify-around w-full mt-4 font-mono text-xs text-slate-600 pt-3 border-t border-slate-100">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-600"></span>
              Real: 20,215 (50.03%)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-rose-600"></span>
              Fake: 20,190 (49.97%)
            </span>
          </div>
        </div>

        {/* Review Length Comparison */}
        <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-2xs lg:col-span-2 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-base font-bold text-slate-900">Review Word Count by Label</h3>
              <span className="text-xs text-slate-500">Average Word Count</span>
            </div>
            <div className="w-full h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={reviewLength} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                  <XAxis dataKey="category" stroke="#64748B" tick={{ fill: '#64748B', fontSize: 12 }} />
                  <YAxis stroke="#64748B" tick={{ fill: '#64748B', fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#FFFFFF',
                      borderColor: '#E2E8F0',
                      color: '#0F172A',
                      borderRadius: '8px',
                      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                      fontSize: '12px',
                    }}
                  />
                  <Legend formatter={(value) => <span className="text-slate-600 text-xs font-medium">{value}</span>} />
                  <Bar dataKey="Real" fill="#059669" radius={[6, 6, 0, 0]} />
                  <Bar dataKey="Fake" fill="#E11D48" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <p className="text-xs text-slate-500 mt-3 pt-3 border-t border-slate-100">
            Real customer reviews are on average ~12 words longer than synthetic GPT-2 generated reviews, revealing structural brevity signatures in machine text.
          </p>
        </div>
      </div>

      {/* Row 2: Category Distribution & Rating Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Category breakdown */}
        <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-2xs">
          <h3 className="text-base font-bold text-slate-900 mb-4">Reviews per Category</h3>
          <div className="w-full h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={categories} layout="vertical" margin={{ top: 10, right: 30, left: 90, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis type="number" stroke="#64748B" tick={{ fill: '#64748B', fontSize: 11 }} />
                <YAxis dataKey="name" type="category" stroke="#64748B" tick={{ fill: '#334155', fontSize: 11 }} width={120} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#FFFFFF',
                    borderColor: '#E2E8F0',
                    color: '#0F172A',
                    borderRadius: '8px',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                    fontSize: '12px',
                  }}
                />
                <Bar dataKey="count" fill="#4F46E5" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Rating Distribution */}
        <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-2xs flex flex-col justify-between">
          <div>
            <h3 className="text-base font-bold text-slate-900 mb-4">Rating Distribution by Label</h3>
            <div className="w-full h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={ratings} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                  <XAxis dataKey="rating" stroke="#64748B" tick={{ fill: '#64748B', fontSize: 12 }} />
                  <YAxis stroke="#64748B" tick={{ fill: '#64748B', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#FFFFFF',
                      borderColor: '#E2E8F0',
                      color: '#0F172A',
                      borderRadius: '8px',
                      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                      fontSize: '12px',
                    }}
                  />
                  <Legend formatter={(value) => <span className="text-slate-600 text-xs font-medium">{value}</span>} />
                  <Bar dataKey="Real" fill="#059669" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Fake" fill="#E11D48" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-xs text-amber-900 flex items-start gap-2.5 mt-2">
            <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
            <span>
              <strong>Key Finding:</strong> Rating distribution is near-identical across both classes - star rating alone provides zero discriminative signal for fake review detection.
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
