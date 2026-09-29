import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Cpu, ShieldCheck, Sparkles, Sliders, Award } from 'lucide-react';

export default function InvestigationTab({ data }) {
  const models = data?.models || [
    { name: 'Engineered Features (Random Forest)', accuracy: 85.27, color: '#64748B' },
    { name: 'TF-IDF (Linear SVM)', accuracy: 87.82, color: '#0284C7' },
    { name: 'Fusion Model (TF-IDF + Engineered)', accuracy: 90.89, color: '#D97706' },
    { name: 'DistilBERT (fine-tuned)', accuracy: 98.05, color: '#4F46E5' },
  ];

  const details = [
    {
      title: 'Lead 1: Hand-Engineered Features',
      icon: Sliders,
      score: '85.27%',
      badgeBg: 'bg-slate-50 border-slate-200 text-slate-700',
      iconBg: 'bg-slate-100 text-slate-700',
      desc: '16 manually derived linguistic, structural, readability, sentiment, and part-of-speech metrics evaluated with Logistic Regression and Random Forest.',
    },
    {
      title: 'Lead 2: TF-IDF Sparse N-Grams',
      icon: Cpu,
      score: '87.82%',
      badgeBg: 'bg-sky-50 border-sky-200 text-sky-700',
      iconBg: 'bg-sky-100 text-sky-700',
      desc: '5,000-dimensional sparse unigram/bigram representation paired with Logistic Regression and Linear Support Vector Machines.',
    },
    {
      title: 'Lead 3: Hybrid Fusion Model',
      icon: Sparkles,
      score: '90.89%',
      badgeBg: 'bg-amber-50 border-amber-200 text-amber-800',
      iconBg: 'bg-amber-100 text-amber-700',
      desc: 'Horizontal concatenation of 5,000 TF-IDF features and 16 scaled engineered features (5,016 total dimensions), calibrated via Linear SVM.',
    },
    {
      title: 'Lead 4: DistilBERT Fine-Tuned Transformer',
      icon: ShieldCheck,
      score: '98.05%',
      badgeBg: 'bg-indigo-50 border-indigo-200 text-indigo-700 font-extrabold',
      iconBg: 'bg-indigo-100 text-indigo-700',
      desc: '6-layer transformer encoder pre-trained on English text and fine-tuned for sequence classification over 3 epochs with AdamW optimizer.',
    },
  ];

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div>
        <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-indigo-50 border border-indigo-100 text-indigo-700 text-xs font-semibold mb-2">
          <Cpu className="w-3.5 h-3.5 text-indigo-600" />
          02 - THE INVESTIGATION & MODELING BENCHMARKS
        </div>
        <h2 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
          Four modeling leads pursued
        </h2>
        <p className="text-sm text-slate-500 mt-1">
          From hand-crafted linguistic features to a fine-tuned transformer encoder, evaluated on the same held-out split.
        </p>
      </div>

      {/* Main Benchmark Chart */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-2xs">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-base font-bold text-slate-900">Model Test Set Accuracy (%)</h3>
            <p className="text-xs text-slate-500">Higher is better - evaluated on 6,061 unseen review samples</p>
          </div>
          <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded font-mono text-xs font-bold bg-emerald-50 text-emerald-700 border border-emerald-200">
            <Award className="w-3.5 h-3.5 text-emerald-600" />
            Top: 98.05%
          </span>
        </div>

        <div className="w-full h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={models}
              layout="vertical"
              margin={{ top: 10, right: 40, left: 190, bottom: 10 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
              <XAxis type="number" domain={[0, 100]} stroke="#64748B" tick={{ fill: '#64748B', fontSize: 12 }} />
              <YAxis
                dataKey="name"
                type="category"
                stroke="#64748B"
                tick={{ fill: '#0F172A', fontSize: 12, fontWeight: 500 }}
                width={180}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#FFFFFF',
                  borderColor: '#E2E8F0',
                  color: '#0F172A',
                  borderRadius: '8px',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                  fontSize: '12px',
                }}
                formatter={(val) => [`${val}%`, 'Accuracy']}
              />
              <Bar dataKey="accuracy" radius={[0, 6, 6, 0]}>
                {models.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Detail Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {details.map((item, idx) => {
          const Icon = item.icon;
          return (
            <div
              key={idx}
              className="bg-white border border-slate-200 rounded-xl p-6 space-y-4 hover:shadow-md transition-all shadow-2xs"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className={`p-2.5 rounded-lg ${item.iconBg}`}>
                    <Icon className="w-5 h-5" />
                  </div>
                  <h4 className="font-bold text-slate-900 text-base">{item.title}</h4>
                </div>
                <span className={`font-mono text-base px-3 py-1 rounded-md border ${item.badgeBg}`}>
                  {item.score}
                </span>
              </div>
              <p className="text-sm text-slate-600 leading-relaxed">
                {item.desc}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
