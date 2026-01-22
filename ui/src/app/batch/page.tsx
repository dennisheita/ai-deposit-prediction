'use client'

import { useState, useEffect } from 'react'

export default function Batch() {
  const [active, setActive] = useState(false)
  const [iterations, setIterations] = useState(0)
  const [mineral, setMineral] = useState('Copper')

  useEffect(() => {
    fetchStatus()
    const interval = setInterval(fetchStatus, 5000)
    return () => clearInterval(interval)
  }, [])

  const fetchStatus = async () => {
    try {
      const response = await fetch('/api/training_status')
      const data = await response.json()
      setActive(data.active)
      setIterations(data.iterations)
    } catch (error) {
      console.error(error)
    }
  }

  const startTraining = async () => {
    try {
      await fetch('/api/start_batch_training', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({ mineral }),
      })
      fetchStatus()
    } catch (error) {
      console.error(error)
    }
  }

  const stopTraining = async () => {
    try {
      await fetch('/api/stop_batch_training', {
        method: 'POST',
      })
      fetchStatus()
    } catch (error) {
      console.error(error)
    }
  }

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold mb-8">Batch Processing</h1>
      <p className="mb-8">Run multiple training jobs with different hyperparameters to find the best model configuration.</p>

      <div className="mb-4">
        <label className="block mb-2">Mineral:</label>
        <select
          value={mineral}
          onChange={(e) => setMineral(e.target.value)}
          className="border p-2"
        >
          <option value="Copper">Copper</option>
          <option value="Diamonds">Diamonds</option>
          <option value="Gold">Gold</option>
          <option value="Lead">Lead</option>
          <option value="REE (Rare Earth Elements)">REE (Rare Earth Elements)</option>
          <option value="Tin">Tin</option>
          <option value="Uranium">Uranium</option>
        </select>
      </div>

      <div className="mb-4">
        <p>Status: {active ? 'Running' : 'Stopped'}</p>
        <p>Iterations: {iterations}</p>
      </div>

      <div className="space-x-4">
        {!active ? (
          <button onClick={startTraining} className="bg-blue-500 text-white px-4 py-2 rounded">
            Start Batch Training
          </button>
        ) : (
          <button onClick={stopTraining} className="bg-red-500 text-white px-4 py-2 rounded">
            Stop Batch Training
          </button>
        )}
      </div>
    </div>
  )
}