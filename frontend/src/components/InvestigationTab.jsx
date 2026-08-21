import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Cpu, ShieldCheck, Sparkles, Sliders } from 'lucide-react';

export default function InvestigationTab({ data }) {
  const models = data?.models || [
    { name: "Engineered Features (Random Forest)", accuracy: 82.75, color: "#8DA0A8" },
    { name: "TF-IDF (Linear SVM)", accuracy: 88.39, color: "#7BB6C9" },
    { name: "Fusion Model (TF-IDF + Engineered)", accuracy: 90.36, color: "#F4C95D" },
    { name: "DistilBERT (fine-tuned)", accuracy: 98.30, color: "#5FD3A0" },
  ];

  const details = [
    {
      title: "Lead 1: Hand-Engineered Features",
      icon: Sliders,
      score: "82.75%",
      color: "border-[#8DA0A8]/40 text-[#8DA0A8]",
      desc: "10 domain features extracted: lexical diversity (TTR), uppercase ratios, sentiment polarity/subjectivity, average word length, punctuation count. Modeled with Random Forest.",
    },
    {
      title: "Lead 2: TF-IDF Sparse N-Grams",
      icon: Cpu,
      score: "88.39%",
      color: "border-[#7BB6C9]/40 text-[#7BB6C9]",
      desc: "Top 100,000 unigram and bigram TF-IDF vectors paired with Linear Support Vector Machines. Captures vocabulary frequency patterns of synthetic text.",
    },
    {
      title: "Lead 3: Hybrid Fusion Model",
      icon: Sparkles,
      score: "90.36%",
      color: "border-[#F4C95D]/40 text-[#F4C95D]",
      desc: "Stitches TF-IDF n-grams with scaled engineered features into a single sparse matrix, calibrated via Logistic Regression and Linear SVM.",
    },
    {
      title: "Lead 4: DistilBERT Fine-Tuned Transformer",
      icon: ShieldCheck,
      score: "98.30%",
      color: "border-[#5FD3A0]/40 text-[#5FD3A0]",
      desc: "6-layer transformer encoder pre-trained on English text and fine-tuned for sequence classification over 3 epochs with AdamW optimizer.",
    },
  ];

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div>
        <div className="inline-flex items-center gap-2 font-mono text-xs text-[#8DA0A8] border border-[#2E3A41] px-2.5 py-1 rounded bg-[#1B2328] mb-2">
          <span className="w-2 h-2 rounded-full bg-[#5FD3A0]"></span>
          02 - THE INVESTIGATION
        </div>
        <h2 className="text-2xl sm:text-3xl font-extrabold text-[#E9EDEE]">
          Four leads pursued
        </h2>
        <p className="text-sm text-[#8DA0A8] mt-1">
          From simple hand-built features to a fine-tuned transformer, each tested fairly on the same held-out data.
        </p>
      </div>

      {/* Main Benchmark Chart */}
      <div className="bg-[#1B2328] border border-[#2E3A41] rounded-xl p-6">
        <h3 className="text-base font-bold text-[#E9EDEE] mb-6">Model Accuracy Comparison (%)</h3>
        <div className="w-full h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={models}
              layout="vertical"
              margin={{ top: 10, right: 50, left: 180, bottom: 10 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#2E3A41" />
              <XAxis type="number" domain={[0, 100]} stroke="#8DA0A8" tick={{ fill: '#8DA0A8', fontSize: 12 }} />
              <YAxis
                dataKey="name"
                type="category"
                stroke="#8DA0A8"
                tick={{ fill: '#E9EDEE', fontSize: 12, fontWeight: 500 }}
                width={170}
              />
              <Tooltip
                contentStyle={{ backgroundColor: '#212B31', borderColor: '#2E3A41', color: '#E9EDEE', borderRadius: '8px' }}
                itemStyle={{ color: '#E9EDEE' }}
                labelStyle={{ color: '#8DA0A8', fontWeight: 600 }}
                cursor={{ fill: 'rgba(255, 255, 255, 0.05)' }}
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

      {/* Detail Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {details.map((item, idx) => {
          const Icon = item.icon;
          return (
            <div key={idx} className="bg-[#1B2328] border border-[#2E3A41] rounded-xl p-6 space-y-4 hover:border-[#8DA0A8]/40 transition-all">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="p-2.5 rounded-lg bg-[#212B31] border border-[#2E3A41]">
                    <Icon className="w-5 h-5 text-[#E9EDEE]" />
                  </div>
                  <h4 className="font-bold text-[#E9EDEE] text-base">{item.title}</h4>
                </div>
                <span className={`font-mono font-extrabold text-lg px-3 py-1 rounded-md border ${item.color}`}>
                  {item.score}
                </span>
              </div>
              <p className="text-sm text-[#8DA0A8] leading-relaxed">
                {item.desc}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
