'use client'

import { useState, useEffect } from 'react'

interface Model {
  0: number
  1: string
  2: string
  3: string
  4: string
}

export default function Compare() {
  const [models, setModels] = useState<Model[]>([])
  const [model1, setModel1] = useState('')
  const [model2, setModel2] = useState('')
  const [mineral, setMineral] = useState('All Minerals')
  const [comparison, setComparison] = useState<{model1: Model | null, model2: Model | null} | null>(null)

  useEffect(() => {
    fetchModels()
  }, [mineral])

  const fetchModels = async () => {
    try {
      const response = await fetch(`/api/models?mineral=${encodeURIComponent(mineral)}`)
      const data = await response.json()
      setModels(data.models)
    } catch (error) {
      console.error(error)
    }
  }

  const handleCompare = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      const response = await fetch('/api/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({
          model1,
          model2,
          mineral,
        }),
      })
      const data = await response.json()
      setComparison(data)
    } catch (error) {
      console.error(error)
    }
  }

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold mb-8">Model Comparison - {mineral}</h1>

      <form onSubmit={handleCompare} className="space-y-4 mb-8">
        <div>
          <label className="block mb-2">Model 1:</label>
          <select
            value={model1}
            onChange={(e) => setModel1(e.target.value)}
            className="border p-2 w-full"
            required
          >
            <option value="">Select Model 1</option>
            {models.map((model) => (
              <option key={model[0]} value={model[0]}>{model[1]}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block mb-2">Model 2:</label>
          <select
            value={model2}
            onChange={(e) => setModel2(e.target.value)}
            className="border p-2 w-full"
            required
          >
            <option value="">Select Model 2</option>
            {models.map((model) => (
              <option key={model[0]} value={model[0]}>{model[1]}</option>
            ))}
          </select>
        </div>
        <button type="submit" className="bg-indigo-500 text-white px-4 py-2 rounded">
          Compare
        </button>
      </form>

      {comparison && (
        <div className="flex space-x-8">
          {comparison.model1 && (
            <div className="flex-1">
              <h2 className="text-2xl font-bold mb-4">Model {comparison.model1[1]}</h2>
              <p>Best Score: {comparison.model1[2]}</p>
              <p>Created: {comparison.model1[3]}</p>
            </div>
          )}
          {comparison.model2 && (
            <div className="flex-1">
              <h2 className="text-2xl font-bold mb-4">Model {comparison.model2[1]}</h2>
              <p>Best Score: {comparison.model2[2]}</p>
              <p>Created: {comparison.model2[3]}</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}