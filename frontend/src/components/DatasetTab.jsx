import React from 'react';
import {
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend,
  BarChart, Bar, XAxis, YAxis, CartesianGrid
} from 'recharts';

export default function DatasetTab({ data }) {
  const classBalance = data?.class_balance || [
    { name: "Real", value: 20215, color: "#5FD3A0" },
    { name: "Fake", value: 20190, color: "#E64980" },
  ];

  const reviewLength = data?.review_length || [
    { category: "avg. words per review", Real: 73.6, Fake: 61.3 }
  ];

  const categories = data?.categories || [
    { name: "Kindle Store", count: 4700 },
    { name: "Books", count: 4400 },
    { name: "Pet Supplies", count: 4250 },
    { name: "Home & Kitchen", count: 4050 },
    { name: "Electronics", count: 4000 },
    { name: "Sports & Outdoors", count: 3950 },
    { name: "Tools & Home Impr.", count: 3900 },
    { name: "Clothing/Shoes/Jewelry", count: 3850 },
    { name: "Toys & Games", count: 3800 },
    { name: "Movies & TV", count: 3600 },
  ];

  const ratings = data?.rating_distribution || [
    { rating: "1★", Real: 1050, Fake: 1050 },
    { rating: "2★", Real: 950, Fake: 950 },
    { rating: "3★", Real: 1850, Fake: 1950 },
    { rating: "4★", Real: 4050, Fake: 3950 },
    { rating: "5★", Real: 12250, Fake: 12250 },
  ];

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div>
        <div className="inline-flex items-center gap-2 font-mono text-xs text-[#8DA0A8] border border-[#2E3A41] px-2.5 py-1 rounded bg-[#1B2328] mb-2">
          <span className="w-2 h-2 rounded-full bg-[#5FD3A0]"></span>
          01 - THE DATASET
        </div>
        <h2 className="text-2xl sm:text-3xl font-extrabold text-[#E9EDEE]">
          40,405 reviews, near-perfectly balanced
        </h2>
        <p className="text-sm text-[#8DA0A8] mt-1">
          10 product categories, real Amazon reviews vs. GPT-2-generated fakes.
        </p>
      </div>

      {/* Row 1: Pie & Length Bar Chart */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Class Balance Pie */}
        <div className="bg-[#1B2328] border border-[#2E3A41] rounded-xl p-5 flex flex-col items-center">
          <h3 className="text-sm font-bold text-[#E9EDEE] w-full text-left mb-4">Class Balance</h3>
          <div className="w-full h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={classBalance}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={90}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {classBalance.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ backgroundColor: '#212B31', borderColor: '#2E3A41', color: '#E9EDEE', borderRadius: '8px' }}
                  itemStyle={{ color: '#E9EDEE' }}
                />
                <Legend formatter={(value) => <span className="text-[#8DA0A8] text-xs font-mono">{value}</span>} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="flex justify-around w-full mt-2 font-mono text-xs text-[#8DA0A8]">
            <span>Real: 20,215 (50.03%)</span>
            <span>Fake: 20,190 (49.97%)</span>
          </div>
        </div>

        {/* Review Length Comparison */}
        <div className="bg-[#1B2328] border border-[#2E3A41] rounded-xl p-5 lg:col-span-2 flex flex-col">
          <h3 className="text-sm font-bold text-[#E9EDEE] mb-4">Review Length by Label (Avg. Words)</h3>
          <div className="w-full h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={reviewLength} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2E3A41" />
                <XAxis dataKey="category" stroke="#8DA0A8" tick={{ fill: '#8DA0A8', fontSize: 12 }} />
                <YAxis stroke="#8DA0A8" tick={{ fill: '#8DA0A8', fontSize: 12 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#212B31', borderColor: '#2E3A41', color: '#E9EDEE', borderRadius: '8px' }}
                />
                <Legend formatter={(value) => <span className="text-[#8DA0A8] text-xs font-mono">{value}</span>} />
                <Bar dataKey="Real" fill="#5FD3A0" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Fake" fill="#E64980" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-xs text-[#8DA0A8] mt-2">
            Real customer reviews are on average ~12 words longer than synthetic GPT-2 generated reviews.
          </p>
        </div>
      </div>

      {/* Row 2: Category Distribution & Rating Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Category breakdown */}
        <div className="bg-[#1B2328] border border-[#2E3A41] rounded-xl p-5">
          <h3 className="text-sm font-bold text-[#E9EDEE] mb-4">Reviews per Category (Approx.)</h3>
          <div className="w-full h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={categories} layout="vertical" margin={{ top: 10, right: 30, left: 80, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2E3A41" />
                <XAxis type="number" stroke="#8DA0A8" tick={{ fill: '#8DA0A8', fontSize: 11 }} />
                <YAxis dataKey="name" type="category" stroke="#8DA0A8" tick={{ fill: '#8DA0A8', fontSize: 11 }} width={110} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#212B31', borderColor: '#2E3A41', color: '#E9EDEE', borderRadius: '8px' }}
                />
                <Bar dataKey="count" fill="#7BB6C9" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Rating Distribution */}
        <div className="bg-[#1B2328] border border-[#2E3A41] rounded-xl p-5 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-bold text-[#E9EDEE] mb-4">Rating Distribution by Label</h3>
            <div className="w-full h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={ratings} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2E3A41" />
                  <XAxis dataKey="rating" stroke="#8DA0A8" tick={{ fill: '#8DA0A8', fontSize: 12 }} />
                  <YAxis stroke="#8DA0A8" tick={{ fill: '#8DA0A8', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#212B31', borderColor: '#2E3A41', color: '#E9EDEE', borderRadius: '8px' }}
                  />
                  <Legend formatter={(value) => <span className="text-[#8DA0A8] text-xs font-mono">{value}</span>} />
                  <Bar dataKey="Real" fill="#5FD3A0" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Fake" fill="#E64980" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="bg-[#212B31] border border-[#2E3A41] rounded-lg p-3 text-xs text-[#8DA0A8] mt-2">
            ⚠️ Rating distribution is near-identical across both classes - star rating alone provides zero discriminative signal for fake detection.
          </div>
        </div>
      </div>
    </div>
  );
}
